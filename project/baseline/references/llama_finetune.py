import json
import logging
import os
from typing import Any, Dict, List

import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LlamaFunctionCallTrainer:
    def __init__(self, model_name: str = "meta-llama/Llama-3.2-1B-Instruct"):
        """
        Llama 3.2 모델을 위한 Function Call Fine-tuning 클래스

        Args:
            model_name: 사용할 Llama 모델명
        """
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.quantized_model = None

    def load_model_and_tokenizer(self):
        """모델과 토크나이저를 불러옵니다."""
        logger.info(f"Loading model and tokenizer: {self.model_name}")

        # 토크나이저 로드
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)

        # 패딩 토큰 설정
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 모델 로드 (16bit로 먼저 로드)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
        )

        # 모델 크기 조정
        self.model.resize_token_embeddings(len(self.tokenizer))

        logger.info("Model and tokenizer loaded successfully")

    def load_function_call_data(self, data_paths: List[str]) -> pd.DataFrame:
        """
        Function Call 데이터를 불러와서 학습용 형태로 변환합니다.

        Args:
            data_paths: csv 파일 경로 리스트

        Returns:
            변환된 데이터 리스트
        """
        logger.info("Loading function call data...")

        all_data = []

        for data_path in data_paths:
            if os.path.exists(data_path):
                df = pd.read_csv(data_path, encoding="utf-8", index_col=0)
                all_data.append(df)
                logger.info(f"Loaded {len(df)} samples from {data_path}")
            else:
                logger.warning(f"Data file not found: {data_path}")

        logger.info(f"Total loaded samples: {len(all_data)}")

        all_data_df = pd.concat(all_data)
        return all_data_df

    def prepare_training_data(self, raw_data: pd.DataFrame) -> Dataset:
        """
        학습 데이터를 준비합니다.

        Args:
            raw_data: 원본 데이터

        Returns:
            학습용 Dataset
        """
        logger.info("Preparing training data...")

        formatted_data = []

        for index, item in raw_data.iterrows():
            # 입력 텍스트 구성
            query = item.get("Query(한글)", "")
            llm_output = item.get("LLM Output", "")

            # 프롬프트 템플릿 구성 (tokenizer.apply_chat_template 활용)
            system_prompt = "당신은 한국어 음성 명령을 Function Call로 변환하는 AI 어시스턴트입니다. 주어진 사용자 명령을 분석하여 적절한 함수 호출 형태로 변환해주세요."
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"사용자 명령: {query}"},
                {"role": "assistant", "content": llm_output},
            ]
            if hasattr(self.tokenizer, "apply_chat_template"):
                prompt = self.tokenizer.apply_chat_template(messages, tokenize=False)
            else:
                # fallback: 기존 프롬프트 방식
                prompt = f"""{system_prompt}\n\n사용자 명령: {query}\n\n{llm_output}"""

            formatted_data.append({"Index": index, "text": prompt, "query": query, "expected_output": llm_output})

        return Dataset.from_list(formatted_data)

    def tokenize_function(self, examples):
        """토크나이징 함수"""
        tokenized = self.tokenizer(
            examples["text"], truncation=True, padding="max_length", max_length=128, return_tensors="pt"
        )
        tokenized["labels"] = tokenized["input_ids"].clone()
        return tokenized

    def setup_lora_config(self) -> LoraConfig:
        """LoRA 설정을 구성합니다."""
        return LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM",
        )

    def fine_tune_model(self, dataset: Dataset, output_dir: str = "./fine_tuned_model", args: Dict[str, Any] = {}):
        """
        모델을 Fine-tune합니다.

        Args:
            dataset: 학습용 데이터셋
            output_dir: 모델 저장 경로
        """
        logger.info("Starting fine-tuning process...")

        # LoRA 설정 적용
        lora_config = self.setup_lora_config()
        self.model = prepare_model_for_kbit_training(self.model)
        self.model = get_peft_model(self.model, lora_config)

        # 데이터셋 토크나이징
        tokenized_dataset = dataset.map(self.tokenize_function, batched=True, remove_columns=dataset.column_names)

        # 학습 인자 설정
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=args.get("num_train_epochs", 3),
            per_device_train_batch_size=args.get("batch_size", 4),
            learning_rate=2e-4,
            fp16=True,
            logging_steps=10,
            save_steps=500,
            eval_strategy="no",
            save_strategy="epoch",
            load_best_model_at_end=False,
            report_to=None,
        )

        # 데이터 콜레이터 설정
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)

        # 트레이너 설정
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )

        # 학습 시작
        logger.info("Starting training...")
        trainer.train()

        # 모델 저장
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)

        logger.info(f"Fine-tuning completed. Model saved to {output_dir}")

    def test_model_inference(self, test_queries: List[str], model_path: str = None):
        """
        모델 추론 테스트를 수행합니다.

        Args:
            test_queries: 테스트할 쿼리 리스트
            model_path: 모델 경로 (None이면 현재 로드된 모델 사용)
        """
        logger.info("Testing model inference...")

        # 모델이 지정되었다면 로드
        if model_path:
            test_model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")
            test_tokenizer = AutoTokenizer.from_pretrained(model_path)
        else:
            test_model = self.model
            test_tokenizer = self.tokenizer

        if test_model is None:
            logger.error("No model loaded for testing")
            return

        for query in test_queries:
            # tokenizer가 chat template 지원 여부에 따라 분기
            system_prompt = "당신은 한국어 음성 명령을 Function Call로 변환하는 AI 어시스턴트입니다. 주어진 사용자 명령을 분석하여 적절한 함수 호출 형태로 변환해주세요."
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"사용자 명령: {query}"},
                {"role": "assistant", "content": ""},
            ]
            if hasattr(test_tokenizer, "apply_chat_template"):
                prompt = test_tokenizer.apply_chat_template(messages, tokenize=False)
            else:
                # fallback: 기존 프롬프트 방식
                prompt = f"{system_prompt}\n\n사용자 명령: {query}\n\n"

            inputs = test_tokenizer(prompt, return_tensors="pt").to(self.model.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=100,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=test_tokenizer.eos_token_id,
                )

            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response.split("assistant<|end_header_id|>")[-1].strip()

            logger.info(f"Query: {query}")
            logger.info(f"Response: {response}")
            logger.info("-" * 50)


if __name__ == "__main__":
    # 사용 예시
    trainer = LlamaFunctionCallTrainer()
    trainer.load_model_and_tokenizer()

    # 데이터 로드
    data_paths = ["/mnt/elice/dataset/train_data.csv"]
    raw_data = trainer.load_function_call_data(data_paths)

    # 학습 데이터 준비
    dataset = trainer.prepare_training_data(raw_data)

    # Fine-tuning 실행
    trainer.fine_tune_model(dataset, output_dir="./my_finetuned_model")
