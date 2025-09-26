import json
import logging
import os
from typing import Any, Dict, List, Union

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LlamaFunctionCallInference:
    def __init__(
        self, model_name: str = "meta-llama/Llama-3.2-1B-Instruct", use_8bit: bool = False, use_4bit: bool = True
    ):
        """
        Llama Function Call 추론 클래스 (베이스라인 - 파인튜닝 없음)

        Args:
            model_name: 사용할 Llama 모델명 (기본값: meta-llama/Llama-3.2-1B-Instruct)
            use_8bit: 8bit 양자화 모델 사용 여부
            use_4bit: 4bit 양자화 모델 사용 여부
        """
        self.model_name = model_name
        self.use_8bit = use_8bit
        self.use_4bit = use_4bit
        self.model = None
        self.tokenizer = None
        self.load_model()

    def load_model(self):
        """모델과 토크나이저를 로드합니다."""
        logger.info(f"Loading model: {self.model_name}")

        # 토크나이저 로드
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True, padding_side="right")

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 모델 로드
        if self.use_8bit:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
            )

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True,
            )
        elif self.use_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
            )

        logger.info("Model loaded successfully")

    def predict_function_call(self, query: Union[str, list[str]]) -> Union[str, list[str]]:
        """
        사용자 쿼리에 대한 Function Call을 예측합니다.
        query가 str이면 단일 예측, list of str이면 배치 예측을 수행합니다.

        Args:
            query: 사용자 명령(str) 또는 사용자 명령 리스트(list of str)

        Returns:
            예측된 Function Call 문자열 또는 리스트
        """
        # query가 str이면 리스트로 변환
        is_single = False
        if isinstance(query, str):
            queries = [query]
            is_single = True
        else:
            queries = query

        # 모델별로 프롬프트 템플릿을 다르게 적용할 수 있도록 일반화
        system_prompt = "당신은 한국어 음성 명령을 Function Call로 변환하는 AI 어시스턴트입니다. 주어진 사용자 명령을 분석하여 적절한 함수 호출 형태로 변환해주세요."

        # 메시지 구조화 (OpenAI/ChatML 스타일 지원)
        messages_list = [
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"사용자 명령: {q}"},
                {"role": "assistant", "content": ""},
            ]
            for q in queries
        ]

        # tokenizer가 chat template 지원 여부에 따라 분기
        if hasattr(self.tokenizer, "apply_chat_template"):
            # chat template 지원 (예: Llama, ChatGLM 등)
            prompts = [self.tokenizer.apply_chat_template(messages, tokenize=False) for messages in messages_list]
        else:
            # 일반 텍스트 프롬프트 (예: GPT2, T5 등)
            prompts = [f"{system_prompt}\n\n사용자 명령: {q}\n\n" for q in queries]

        # 토크나이징 (batch)
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.model.device)

        # 추론 수행 (batch)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.1,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1,
            )

        # 응답 디코딩 (batch)
        responses = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        results = []
        for response in responses:
            resp = response.split("assistant")[-1].strip()
            resp = resp.split("system")[-1].strip()
            resp = resp.split("user")[-1].strip()
            if "<|eot_id|>" in resp:
                resp = resp.split("<|eot_id|>")[0].strip()
            if "<end>" in resp:
                resp = resp.split("<end>")[0].strip() + "<end>"
            results.append(resp)

        # 입력이 단일 str이면 str로 반환, 아니면 list로 반환
        if is_single:
            return results[0]
        else:
            return results

    def batch_predict(self, queries: List[str], batch_size: int = 64) -> List[str]:
        """
        여러 쿼리에 대한 배치 예측을 수행합니다.

        Args:
            queries: 사용자 명령 리스트
            batch_size: 배치 크기

        Returns:
            예측된 Function Call 리스트
        """
        results = []

        for start in tqdm(range(0, len(queries), batch_size)):
            end = min(start + batch_size, len(queries))
            if end >= len(queries):
                batch_queries = queries[start:]
            else:
                batch_queries = queries[start:end]
            batch_results = self.predict_function_call(batch_queries)
            results.extend(batch_results)

        return results

    def evaluate_model(self, test_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        모델 성능을 평가합니다.

        Args:
            test_data: 테스트 데이터

        Returns:
            평가 결과 딕셔너리
        """
        correct = 0
        total = len(test_data)

        for item in test_data:
            query = item.get("Query(한글)", "")
            expected_output = item.get("LLM Output", "")

            predicted_output = self.predict_function_call(query)

            if predicted_output.strip() == expected_output.strip():
                correct += 1

        accuracy = correct / total if total > 0 else 0

        return {"accuracy": accuracy, "correct": correct, "total": total}


def main():
    """메인 실행 함수"""
    # 베이스라인 모델로 추론 테스트
    inference = LlamaFunctionCallInference()

    # 테스트 쿼리
    test_queries = [
        "공기 청정기 켜줘",
        "스테이션으로 돌아가",
        "풍량 강풍으로 바꿔줘",
        "청정 기능 켜줘",
        "터보로 바꿔줘",
        "충전 해줘",
    ]

    # 개별 예측
    logger.info("=== 개별 예측 테스트 ===")
    for query in test_queries:
        result = inference.predict_function_call(query)
        logger.info(f"Query: {query}")
        logger.info(f"Prediction: {result}")
        logger.info("-" * 50)


if __name__ == "__main__":
    main()
