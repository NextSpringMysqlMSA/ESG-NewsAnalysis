import json
import os
import logging
from kafka import KafkaConsumer, KafkaProducer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import json
import logging
import os
from datetime import datetime
import gc
import torch

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelKafka:
    def __init__(self,
                 bootstrap_servers=os.environ.get('KAFKA_BROKERS', 'localhost:9092'),
                 consume_topic=os.environ.get('KAFKA_CONSUME_TOPIC', "news-results"),
                 produce_topic=os.environ.get('KAFKA_PRODUCE_TOPIC', "esg-analysis-results"),
                 group_id=os.environ.get('KAFKA_GROUP_ID', 'esg-analysis-group'),
                 translate_model_name="facebook/nllb-200-distilled-600M",
                 sentiment_model_name="snunlp/KR-FinBert-SC",
                 esg_model_name="yiyanghkust/finbert-esg",
                 device=-1): # CPU 사용: -1, GPU 사용: 0 (또는 특정 GPU 인덱스)

        self.bootstrap_servers = bootstrap_servers
        self.consume_topic = consume_topic
        self.produce_topic = produce_topic
        self.group_id = group_id
        self.device = device
        
        logger.info(f"Kafka Bootstrap Servers: {self.bootstrap_servers}")
        logger.info(f"Kafka Consume Topic: {self.consume_topic}")
        logger.info(f"Kafka Produce Topic: {self.produce_topic}")
        logger.info(f"Kafka Group ID: {self.group_id}")
        logger.info(f"Using device: {'GPU' if device >= 0 else 'CPU'}")

        # 메모리 최적화를 위한 설정
        if device == -1:  # CPU 사용 시
            torch.set_num_threads(2)  # CPU 스레드 수 제한
        
        # 모델 로드 (애플리케이션 시작 시 1회)
        self._load_models(translate_model_name, sentiment_model_name, esg_model_name)
        self._setup_kafka()

    def _load_models(self, translate_model_name, sentiment_model_name, esg_model_name):
        """모델들을 로드하고 메모리 사용량을 모니터링합니다."""
        try:
            logger.info(f"Loading translation model: {translate_model_name}")
            self.trans_tokenizer = AutoTokenizer.from_pretrained(translate_model_name)
            self.trans_model = AutoModelForSeq2SeqLM.from_pretrained(translate_model_name)
            self.src_lang = "kor_Kore"
            self.tgt_lang = "eng_Latn"
            
            # GPU 사용 시 모델을 GPU로 이동
            if self.device >= 0:
                self.trans_model = self.trans_model.to(f'cuda:{self.device}')

            logger.info(f"Loading sentiment analysis model: {sentiment_model_name}")
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=sentiment_model_name,
                tokenizer=sentiment_model_name,
                device=self.device
            )

            logger.info(f"Loading ESG classification model: {esg_model_name}")
            self.esg_pipeline = pipeline(
                "text-classification",
                model=esg_model_name,
                tokenizer=esg_model_name,
                device=self.device
            )
            
            # NLLB 모델의 forced_bos_token_id 설정
            self.forced_bos_token_id = self.trans_tokenizer.convert_tokens_to_ids(f"{self.tgt_lang}")
            
            # 메모리 정리
            gc.collect()
            if self.device >= 0:
                torch.cuda.empty_cache()
                
            logger.info("All models loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            raise

    def _setup_kafka(self):
        """Kafka Consumer와 Producer를 설정합니다."""
        # Consumer 설정
        try:
            self.consumer = KafkaConsumer(
                self.consume_topic,
                bootstrap_servers=self.bootstrap_servers.split(','),
                auto_offset_reset='earliest',
                group_id=self.group_id,
                enable_auto_commit=True,
                value_deserializer=lambda x: json.loads(x.decode('utf-8')),
                consumer_timeout_ms=1000,
                max_poll_records=5,  # 배치 크기 제한
                session_timeout_ms=30000,  # 세션 타임아웃 설정
                heartbeat_interval_ms=10000  # 하트비트 간격 설정
            )
            logger.info("Kafka Consumer initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize Kafka Consumer: {e}")
            raise

        # Producer 설정
        try:
            self.producer = KafkaProducer(
                bootstrap_servers=self.bootstrap_servers.split(','),
                value_serializer=lambda m: json.dumps(m, ensure_ascii=False).encode('utf-8'),
                batch_size=16384,  # 배치 크기 설정
                linger_ms=10,  # 배치 대기 시간
                compression_type='gzip'  # 압축 사용
            )
            logger.info("Kafka Producer initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize Kafka Producer: {e}")
            raise

    def translate(self, text: str) -> str:
        """텍스트를 한국어에서 영어로 번역합니다."""
        if not text or not text.strip():
            return ""
        
        try:
            # 텍스트 길이 제한
            if len(text) > 1000:
                text = text[:1000]
                logger.warning("Text truncated to 1000 characters for translation")
            
            encoded = self.trans_tokenizer(
                text, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=512
            )
            
            # GPU 사용 시 데이터를 GPU로 이동
            if self.device >= 0:
                encoded = {k: v.to(f'cuda:{self.device}') for k, v in encoded.items()}
            
            with torch.no_grad():  # 메모리 절약을 위해 gradient 계산 비활성화
                generated_tokens = self.trans_model.generate(
                    **encoded,
                    forced_bos_token_id=self.forced_bos_token_id,
                    max_length=512,
                    num_beams=1,  # 빠른 생성을 위해 beam search 비활성화
                    do_sample=False
                )
            
            result = self.trans_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
            
            # 메모리 정리
            del encoded, generated_tokens
            if self.device >= 0:
                torch.cuda.empty_cache()
                
            return result.strip()
            
        except Exception as e:
            logger.error(f"Error during translation: {e}")
            # 번역 실패 시 원본 텍스트 반환 (ESG 분석이 스킵되지 않도록)
            return text

    def _analyze_sentiment_and_esg(self, text_to_analyze, translated_text):
        """감성 분석과 ESG 분류를 수행합니다."""
        sentiment_result = None
        esg_result = None
        
        try:
            # 감성 분석 (한국어 텍스트 사용)
            if text_to_analyze:
                sentiment_output = self.sentiment_pipeline(text_to_analyze)
                sentiment_result = sentiment_output[0] if isinstance(sentiment_output, list) else sentiment_output
                logger.debug(f"Sentiment analysis completed: {sentiment_result}")
        except Exception as e:
            logger.error(f"Error during sentiment analysis: {e}")
            sentiment_result = {"label": "UNKNOWN", "score": 0.0}

        try:
            # ESG 분류 (번역된 영어 텍스트 사용)
            if translated_text and translated_text.strip():
                esg_output = self.esg_pipeline(translated_text)
                esg_result = esg_output[0] if isinstance(esg_output, list) else esg_output
                logger.debug(f"ESG classification completed: {esg_result}")
        except Exception as e:
            logger.error(f"Error during ESG classification: {e}")
            esg_result = {"label": "UNKNOWN", "score": 0.0}

        return sentiment_result, esg_result

    def consume_messages(self):
        logger.info("Starting to consume messages...")
        processed_count = 0
        
        try:
            while True:
                msg_pack = self.consumer.poll(timeout_ms=1000, max_records=5)

                if not msg_pack:
                    continue

                for tp, messages in msg_pack.items():
                    for message in messages:
                        try:
                            self._process_single_message(message, tp)
                            processed_count += 1
                            
                            # 주기적으로 메모리 정리
                            if processed_count % 10 == 0:
                                gc.collect()
                                if self.device >= 0:
                                    torch.cuda.empty_cache()
                                logger.debug(f"Processed {processed_count} messages, memory cleaned")
                                
                        except Exception as e:
                            logger.error(f"Error processing individual message: {e}", exc_info=True)
                            
        except KeyboardInterrupt:
            logger.info("Consumer interrupted by KeyboardInterrupt. Closing...")
        except Exception as e:
            logger.critical(f"An unexpected error occurred in consume_messages: {e}", exc_info=True)
        finally:
            self._cleanup_resources()

    def _process_single_message(self, message, tp):
        """단일 메시지를 처리합니다."""
        message_value = message.value 
        logger.info(f"Received message from topic {tp.topic} partition {tp.partition} offset {message.offset}")

        original_keyword = message_value.get("keyword", "N/A")
        news_items = message_value.get("newsItems", [])

        if not news_items:
            logger.warning("No newsItems found in the message.")
            return

        for news_item in news_items:
            try:
                title = news_item.get("title", "")
                summary = news_item.get("description", "")  # description 필드 사용
                news_url = news_item.get("url", "")  # 실제 뉴스 URL
                original_url = news_item.get("originalUrl", "")  # 원문 URL
                published_at = news_item.get("publishedAt", "")
                
                if not title and not summary:
                    logger.warning(f"Skipping item with no title and summary: {news_item}")
                    continue
                    
                text_to_analyze = f"{title}: {summary}" if title and summary else title or summary
                translated_text = self.translate(text_to_analyze)
                
                sentiment_result, esg_result = self._analyze_sentiment_and_esg(text_to_analyze, translated_text)

                # 뉴스 URL 우선순위: originalUrl > url
                final_news_url = original_url if original_url else news_url

                analysis_result = {
                    "original_keyword": original_keyword,
                    "original_news_item": text_to_analyze,
                    "news_url": final_news_url,  # 실제 뉴스 URL 추가
                    "news_title": title,  # 뉴스 제목 추가
                    "published_at": published_at,  # 발행 시간 추가
                    "sentiment": sentiment_result,
                    "esg_classification": esg_result,
                    "analysis_timestamp": datetime.now().isoformat()
                }
                
                self.producer.send(self.produce_topic, value=analysis_result)
                logger.info(f"Sent analysis result to Kafka topic '{self.produce_topic}' with URL: {final_news_url}")

            except Exception as e:
                logger.error(f"Error processing news_item: {e}", exc_info=True)

    def _cleanup_resources(self):
        """리소스를 정리합니다."""
        logger.info("Cleaning up resources...")
        
        try:
            if hasattr(self, 'producer') and self.producer:
                self.producer.flush()
                self.producer.close()
                
            if hasattr(self, 'consumer') and self.consumer:
                self.consumer.close()
                
            # 메모리 정리
            gc.collect()
            if self.device >= 0:
                torch.cuda.empty_cache()
                
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
        finally:
            logger.info("Resource cleanup completed.")

if __name__ == "__main__":
    device_option = -1  # 기본적으로 CPU 사용
    
    # GPU 사용 여부 환경변수로 설정 가능
    if os.environ.get("USE_GPU", "false").lower() == "true":
        device_option = 0

    try:
        model_kafka = ModelKafka(device=device_option)
        model_kafka.consume_messages()
    except Exception as e:
        logger.critical(f"Application failed to start or run: {e}", exc_info=True)

# 사용된 모델 이름:
# 번역: facebook/nllb-200-distilled-600M
# 감성분석: snunlp/KR-FinBert-SC
# ESG 분류: yiyanghkust/finbert-esg
# (만약 다른 모델을 사용하려면 __init__의 기본값을 변경하거나 환경변수 등으로 주입하세요)
#
# 주요 변경사항:
# 1. Kafka 설정 (bootstrap_servers, consume_topic, produce_topic, group_id)을 환경 변수에서 읽도록 수정.
# 2. newsItems 배열의 모든 아이템을 순회하며 처리.
# 3. logging 모듈을 사용하여 상세 로깅 추가.
# 4. 모델 로드를 클래스 생성자(__init__)로 이동하여 한 번만 실행되도록 함.
# 5. pipeline 생성 시 device 파라미터를 추가하여 CPU/GPU 사용을 명시적으로 제어 (기본 CPU).
# 6. Kafka Producer가 JSON 객체를 직렬화하도록 수정.
# 7. 각 뉴스 아이템 처리 중 발생하는 오류를 로깅하고, 다음 아이템 처리를 계속하도록 함.
# 8. NLLB 번역 모델 사용 시 forced_bos_token_id 설정 방식을 명확히 함.
# 9. Kafka consumer/producer 초기화 실패 시 에러 발생 및 로깅.
#10. consumer_timeout_ms 추가하여 무한 대기 방지 및 루프 제어.
#11. 분석 결과에 원본 뉴스 아이템과 키워드, 분석 시간 등을 포함.

    
