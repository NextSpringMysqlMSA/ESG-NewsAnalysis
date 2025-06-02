import json
import os
import logging
from kafka import KafkaConsumer, KafkaProducer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import json
import logging
import os
from datetime import datetime

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
        
        logger.info(f"Kafka Bootstrap Servers: {self.bootstrap_servers}")
        logger.info(f"Kafka Consume Topic: {self.consume_topic}")
        logger.info(f"Kafka Produce Topic: {self.produce_topic}")
        logger.info(f"Kafka Group ID: {self.group_id}")

        # 모델 로드 (애플리케이션 시작 시 1회)
        logger.info(f"Loading translation model: {translate_model_name}")
        self.trans_tokenizer = AutoTokenizer.from_pretrained(translate_model_name)
        self.trans_model = AutoModelForSeq2SeqLM.from_pretrained(translate_model_name)
        self.src_lang = "kor_Kore"
        self.tgt_lang = "eng_Latn"
        # 번역 모델에 소스 언어 설정 (필요한 경우)
        # self.trans_model.config.forced_bos_token_id = self.trans_tokenizer.lang_code_to_id[self.tgt_lang] # NLLB 모델의 경우

        logger.info(f"Loading sentiment analysis model: {sentiment_model_name}")
        self.sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model=sentiment_model_name,
            tokenizer=sentiment_model_name,
            device=device
        )

        logger.info(f"Loading ESG classification model: {esg_model_name}")
        self.esg_pipeline = pipeline(
            "text-classification",
            model=esg_model_name,
            tokenizer=esg_model_name,
            device=device
        )

        # Consumer 설정
        try:
            self.consumer = KafkaConsumer(
                self.consume_topic,
                bootstrap_servers=self.bootstrap_servers.split(','), # 여러 브로커를 위해 split
                auto_offset_reset='earliest',
                group_id=self.group_id,
                enable_auto_commit=True,
                value_deserializer=lambda x: json.loads(x.decode('utf-8')),
                consumer_timeout_ms=1000 # 폴링 타임아웃 (ms)
            )
            logger.info("Kafka Consumer initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize Kafka Consumer: {e}")
            raise

        # Producer 설정
        try:
            self.producer = KafkaProducer(
                bootstrap_servers=self.bootstrap_servers.split(','), # 여러 브로커를 위해 split
                value_serializer=lambda m: json.dumps(m, ensure_ascii=False).encode('utf-8') # JSON 객체를 직렬화
            )
            logger.info("Kafka Producer initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize Kafka Producer: {e}")
            raise
            
        # NLLB 모델의 forced_bos_token_id 설정
        # NLLB-200 모델들은 일반적으로 target language code를 tokenizer에 전달하여 생성합니다.
        # pipeline 사용 시 내부적으로 처리될 수 있으나, 직접 generate를 사용할 경우 필요합니다.
        # 여기서는 translate 함수에서 직접 generate를 사용하므로 설정합니다.
        self.forced_bos_token_id = self.trans_tokenizer.convert_tokens_to_ids(f"{self.tgt_lang}") # NLLB-200 스타일
        # 만약 `facebook/nllb-200-distilled-600M` 모델이 `<<eng_Latn>>` 형식을 사용한다면 아래와 같이 수정
        # self.forced_bos_token_id = self.trans_tokenizer.convert_tokens_to_ids(f"<<{self.tgt_lang}>>")


    def translate(self, text: str) -> str:
        if not text:
            return ""
        try:
            encoded = self.trans_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            # GPU 사용 시 모델과 데이터를 동일 장치로 이동
            # if self.sentiment_pipeline.device.type == 'cuda':
            #    encoded = {k: v.to(self.sentiment_pipeline.device) for k, v in encoded.items()}
            
            generated_tokens = self.trans_model.generate(
                **encoded,
                forced_bos_token_id=self.forced_bos_token_id,
                max_length=512 # 요약문의 길이에 따라 조정 가능
            )
            result = self.trans_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
            return result
        except Exception as e:
            logger.error(f"Error during translation: {e}")
            return "" # 번역 실패 시 빈 문자열 반환

    def consume_messages(self):
        logger.info("Starting to consume messages...")
        try:
            while True: # 무한 루프로 변경하여 계속 메시지를 기다림
                # poll 메서드로 메시지를 가져옴. 타임아웃(ms) 설정 가능.
                # 반환값은 {TopicPartition: [ConsumerRecord, ...]} 형태의 딕셔너리
                msg_pack = self.consumer.poll(timeout_ms=1000, max_records=5) # 예: 1초마다 최대 5개 레코드 배치 확인

                if not msg_pack: # 타임아웃 동안 메시지가 없으면 다음 폴링 시도
                    continue

                for tp, messages in msg_pack.items():
                    for message in messages: # ConsumerRecord
                        # message.value는 이미 __init__에서 설정한 value_deserializer에 의해 dict로 변환됨
                        message_value = message.value 
                        logger.info(f"Received message from topic {tp.topic} partition {tp.partition} offset {message.offset}: {message_value}")

                        original_keyword = message_value.get("keyword", "N/A")
                        news_items = message_value.get("newsItems", [])

                        if not news_items:
                            logger.warning("No newsItems found in the message.")
                            continue

                        for news_item in news_items:
                            try:
                                title = news_item.get("title", "")
                                summary = news_item.get("summary", "")
                                # link = news_item.get("link", "") # 현재 사용 안 함
                                # timestamp = news_item.get("timestamp", "") # 현재 사용 안 함
                                # source = news_item.get("source", "") # 현재 사용 안 함
                                
                                if not title and not summary:
                                    logger.warning(f"Skipping item with no title and summary: {news_item}")
                                    continue
                                    
                                text_to_analyze = f"{title}: {summary}" if title and summary else title or summary

                                sentiment_result = None
                                if text_to_analyze:
                                    # 파이프라인 직접 호출 시 리스트로 반환될 수 있으므로 첫 번째 요소 사용 가정
                                    sentiment_output = self.sentiment_pipeline(text_to_analyze)
                                    sentiment_result = sentiment_output[0] if isinstance(sentiment_output, list) else sentiment_output
                                    logger.info(f"Sentiment analysis for '{text_to_analyze[:50]}...': {sentiment_result}")

                                translated_text = self.translate(text_to_analyze)
                                logger.info(f"Translated text for ESG analysis: {translated_text[:50]}...")
                                
                                esg_result = None
                                if translated_text: # 번역된 텍스트가 있을 경우에만 ESG 분석 수행
                                    # 파이프라인 직접 호출 시 리스트로 반환될 수 있으므로 첫 번째 요소 사용 가정
                                    esg_output = self.esg_pipeline(translated_text)
                                    esg_result = esg_output[0] if isinstance(esg_output, list) else esg_output
                                    logger.info(f"ESG classification for '{translated_text[:50]}...': {esg_result}")

                                analysis_result = {
                                    "original_keyword": original_keyword,
                                    "original_news_item": news_item, # 원본 뉴스 아이템 포함
                                    "sentiment": sentiment_result,
                                    "esg_classification": esg_result,
                                    "analysis_timestamp": logging.Formatter().formatTime(logging.LogRecord(None,None,"",0,"", (), None, None)) # 현재 시간 ISO
                                }
                                
                                self.producer.send(self.produce_topic, value=analysis_result)
                                self.producer.flush() # 개발 중에는 즉시 전송, 운영 시에는 조절 가능
                                logger.info(f"Sent analysis result to Kafka topic '{self.produce_topic}': {analysis_result}")

                            except Exception as e:
                                logger.error(f"Error processing news_item {news_item.get('link', 'N/A')}: {e}", exc_info=True)
                                # 실패한 메시지에 대한 처리 (예: DLQ 전송)는 여기에 추가 가능
        except KeyboardInterrupt:
            logger.info("Consumer interrupted by KeyboardInterrupt. Closing...")
        except Exception as e:
            logger.critical(f"An unexpected error occurred in consume_messages: {e}", exc_info=True)
        finally:
            logger.info("Closing Kafka consumer and producer due to loop exit or error...")
            if hasattr(self, 'consumer') and self.consumer:
                self.consumer.close()
            if hasattr(self, 'producer') and self.producer:
                self.producer.close()
            logger.info("Kafka consumer and producer resources released.")


if __name__ == "__main__":
    # GPU 사용 여부 결정 (환경 변수 또는 설정 파일 등에서 읽어올 수 있음)
    # 예: USE_GPU = os.environ.get("USE_GPU", "false").lower() == "true"
    # device_option = 0 if USE_GPU else -1
    device_option = -1 # 기본적으로 CPU 사용

    # 환경 변수에서 모델 이름 읽기 (선택 사항)
    # translate_model = os.environ.get("TRANSLATE_MODEL_NAME", "facebook/nllb-200-distilled-600M")
    # sentiment_model = os.environ.get("SENTIMENT_MODEL_NAME", "snunlp/KR-FinBert-SC")
    # esg_model = os.environ.get("ESG_MODEL_NAME", "yiyanghkust/finbert-esg")

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

    
