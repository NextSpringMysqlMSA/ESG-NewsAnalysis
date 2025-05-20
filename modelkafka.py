from kafka import KafkaConsumer, KafkaProducer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import pipeline

import json

class ModelKafka:
    def __init__(self, bootstrap_servers = '192.168.0.224:9092', 
                 consume_topic = "news-results", produce_topic = "output-topic",
                 translate_model = "facebook/nllb-200-distilled-600M",
                 sentiment_model = "snunlp/KR-FinBert-SC", esg_model = "yiyanghkust/finbert-esg"):
        self.bootstrap_servers = bootstrap_servers
        self.consume_topic = consume_topic  # 소비할 토픽
        self.produce_topic = produce_topic
        self.translate_model = translate_model
        self.trans_tokenizer = AutoTokenizer.from_pretrained(self.translate_model)
        self.trans_model = AutoModelForSeq2SeqLM.from_pretrained(self.translate_model)
        self.src_lang = "kor_Kore"
        self.tgt_lang = "eng_Latn"
        self.sentiment = pipeline(
        "sentiment-analysis",
        model = sentiment_model,  # 또는 "nlpai-lab/kcbert-base-sentiment"
        tokenizer = sentiment_model, # GPU 사용 가능 시
        )
        # Consumer 설정
        self.consumer = KafkaConsumer(
            self.consume_topic,
            bootstrap_servers=self.bootstrap_servers,
            auto_offset_reset='earliest',
            group_id='python-test-group',
            enable_auto_commit=True,
            value_deserializer=lambda x: json.loads(x.decode('utf-8'))  # JSON 디코딩
        )

        # Producer 설정
        self.producer = KafkaProducer(
            bootstrap_servers=self.bootstrap_servers,
            value_serializer=lambda m: m.encode('utf-8')
        )
        self.trans_model.src_lang = self.src_lang
        self.forced_bos_token_id=self.trans_tokenizer.convert_tokens_to_ids(f"<<{self.tgt_lang}>>")
        self.esg = pipeline("text-classification", model=esg_model, tokenizer=esg_model)  # GPU 사용 가능 
        self.consume_messages()
    def consume_messages(self):
        for message in self.consumer:
            send_data = {}
            message_value = message.value
            print("[수신 데이터]", message_value)

            try:
                title = message_value['newsItems'][0]["title"]
                summary = message_value['newsItems'][0]["summary"]
                send_data['receive'] = message_value
                send_data['sentiment'] = self.sentiment(f"{title}: {summary}")
                
                trans = self.translate(summary)
                send_data['esg'] = self.esg(trans)

                # 전송용 JSON 문자열로 직렬화
                json_data = json.dumps(send_data, ensure_ascii=False)
                self.producer.send(self.produce_topic, value=json_data)
                self.producer.flush()  # 즉시 전송 (옵션)

                print("[전송 데이터]", json_data)
            except Exception as e:
                print(f"[오류 발생]: {e}")

    def translate(self,text):

        encoded = self.trans_tokenizer(text, return_tensors="pt", padding=True, truncation=True)

        # ✅ 번역 수행
        generated_tokens = self.trans_model.generate(
            **encoded,
            forced_bos_token_id=self.forced_bos_token_id,
            max_length=512
        )

        # ✅ 결과 디코딩
        result = self.trans_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        return result

if __name__ == "__main__":
    model_kafka = ModelKafka()
    
