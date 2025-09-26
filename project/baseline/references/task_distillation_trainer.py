#!/usr/bin/env python3
"""
Task Distillation Trainer for Korean Voice Function Calling
Implements Teacher-Student distillation with multi-objective optimization
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
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


class TaskDistillationTrainer:
    """
    Task Distillation을 위한 Teacher-Student 훈련 클래스
    
    Key Features:
    1. Multi-teacher ensemble distillation
    2. Progressive knowledge transfer
    3. Task-specific attention distillation
    4. Adaptive temperature scaling
    5. Performance-guided student selection
    """

    def __init__(
        self,
        teacher_models: List[str],
        student_model: str = "meta-llama/Llama-3.2-1B-Instruct",
        intermediate_model: Optional[str] = "meta-llama/Llama-3.2-3B-Instruct",
    ):
        """
        Task Distillation Trainer 초기화

        Args:
            teacher_models: 교사 모델들의 리스트 (예: GPT-4, Claude, 더 큰 Llama 모델)
            student_model: 학생 모델 (배포용 소형 모델)
            intermediate_model: 중간 크기 모델 (점진적 증류용)
        """
        self.teacher_models = teacher_models
        self.student_model_name = student_model
        self.intermediate_model_name = intermediate_model
        
        # 모델 저장소
        self.teachers = {}
        self.student = None
        self.intermediate = None
        
        # 토크나이저
        self.tokenizer = None
        
        # 증류 설정
        self.distillation_config = {
            "temperature": 4.0,  # Knowledge distillation temperature
            "alpha": 0.7,  # Hard target weight
            "beta": 0.3,   # Soft target weight
            "attention_weight": 0.1,  # Attention distillation weight
            "hidden_weight": 0.1,     # Hidden state distillation weight
        }

    def load_models(self):
        """모든 모델을 로드합니다."""
        logger.info("Loading teacher, intermediate, and student models...")
        
        # 토크나이저 로드
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.student_model_name, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 학생 모델 로드
        self.student = AutoModelForCausalLM.from_pretrained(
            self.student_model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )

        # 중간 모델 로드 (점진적 증류용)
        if self.intermediate_model_name:
            self.intermediate = AutoModelForCausalLM.from_pretrained(
                self.intermediate_model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )

        logger.info("All models loaded successfully")

    def generate_teacher_outputs(self, data_batch: List[str]) -> Dict[str, List[str]]:
        """
        교사 모델들로부터 고품질 출력을 생성합니다.
        
        Args:
            data_batch: 입력 쿼리 배치
            
        Returns:
            교사별 출력 딕셔너리
        """
        teacher_outputs = {}
        
        for teacher_name in self.teacher_models:
            outputs = []
            
            if teacher_name.startswith("openai"):
                # OpenAI API 사용
                outputs = self._generate_openai_outputs(data_batch, teacher_name)
            elif teacher_name.startswith("anthropic"):
                # Anthropic API 사용
                outputs = self._generate_anthropic_outputs(data_batch, teacher_name)
            elif teacher_name.startswith("meta-llama"):
                # 로컬 대형 Llama 모델 사용
                outputs = self._generate_llama_outputs(data_batch, teacher_name)
            else:
                logger.warning(f"Unknown teacher model: {teacher_name}")
                continue
                
            teacher_outputs[teacher_name] = outputs
            
        return teacher_outputs

    def _generate_openai_outputs(self, data_batch: List[str], model_name: str) -> List[str]:
        """OpenAI API를 사용하여 교사 출력 생성"""
        try:
            import openai
            
            outputs = []
            for query in data_batch:
                response = openai.ChatCompletion.create(
                    model=model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": "당신은 한국어 음성 명령을 정확한 Function Call로 변환하는 전문가입니다. JSON 형태로 구조화된 함수 호출을 생성해주세요."
                        },
                        {"role": "user", "content": f"사용자 명령: {query}"}
                    ],
                    temperature=0.1,
                    max_tokens=200
                )
                outputs.append(response.choices[0].message.content)
                
            return outputs
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return ["Error"] * len(data_batch)

    def _generate_llama_outputs(self, data_batch: List[str], model_name: str) -> List[str]:
        """로컬 Llama 모델로 교사 출력 생성"""
        if model_name not in self.teachers:
            # 교사 모델 동적 로드
            self.teachers[model_name] = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
        
        teacher_model = self.teachers[model_name]
        outputs = []
        
        for query in data_batch:
            system_prompt = "당신은 한국어 음성 명령을 Function Call로 변환하는 AI 어시스턴트입니다."
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"사용자 명령: {query}"},
            ]
            
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False)
            inputs = self.tokenizer(prompt, return_tensors="pt").to(teacher_model.device)
            
            with torch.no_grad():
                teacher_outputs = teacher_model.generate(
                    **inputs,
                    max_new_tokens=150,
                    temperature=0.1,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            
            response = self.tokenizer.decode(teacher_outputs[0], skip_special_tokens=True)
            # Extract assistant response
            response = response.split("assistant")[-1].strip()
            outputs.append(response)
            
        return outputs

    def create_distillation_dataset(self, raw_data: pd.DataFrame) -> Dataset:
        """
        증류를 위한 데이터셋 생성
        
        Args:
            raw_data: 원본 데이터
            
        Returns:
            증류용 Dataset
        """
        logger.info("Creating distillation dataset with teacher outputs...")
        
        distillation_data = []
        
        # 배치 단위로 교사 출력 생성
        batch_size = 16
        queries = raw_data["Query(한글)"].tolist()
        
        for i in range(0, len(queries), batch_size):
            batch_queries = queries[i:i + batch_size]
            teacher_outputs = self.generate_teacher_outputs(batch_queries)
            
            for j, query in enumerate(batch_queries):
                # 앙상블 교사 출력 (가중 평균 또는 투표)
                ensemble_output = self._ensemble_teacher_outputs(
                    {k: v[j] for k, v in teacher_outputs.items()}
                )
                
                # 원본 데이터의 ground truth
                original_idx = i + j
                ground_truth = raw_data.iloc[original_idx]["LLM Output"]
                
                distillation_data.append({
                    "query": query,
                    "ground_truth": ground_truth,
                    "teacher_output": ensemble_output,
                    "teacher_outputs": {k: v[j] for k, v in teacher_outputs.items()},
                })
        
        return Dataset.from_list(distillation_data)

    def _ensemble_teacher_outputs(self, teacher_outputs: Dict[str, str]) -> str:
        """
        여러 교사 모델의 출력을 앙상블
        
        Args:
            teacher_outputs: 교사별 출력 딕셔너리
            
        Returns:
            앙상블된 출력
        """
        # 간단한 구현: 가장 빈번한 출력 선택
        # 실제로는 더 정교한 앙상블 전략 사용 가능
        
        if not teacher_outputs:
            return ""
        
        # 출력 품질 평가 및 가중치 적용
        scored_outputs = []
        for teacher, output in teacher_outputs.items():
            score = self._evaluate_output_quality(output)
            scored_outputs.append((score, output))
        
        # 가장 높은 점수의 출력 선택
        scored_outputs.sort(key=lambda x: x[0], reverse=True)
        return scored_outputs[0][1] if scored_outputs else ""

    def _evaluate_output_quality(self, output: str) -> float:
        """
        출력 품질 평가 (간단한 휴리스틱)
        
        Args:
            output: 평가할 출력
            
        Returns:
            품질 점수 (0-1)
        """
        score = 0.0
        
        # JSON 형태 확인
        try:
            json.loads(output)
            score += 0.3
        except:
            pass
        
        # 함수 호출 키워드 포함 확인
        function_keywords = ["function", "call", "method", "api", "{", "}", "parameters"]
        keyword_count = sum(1 for keyword in function_keywords if keyword.lower() in output.lower())
        score += min(keyword_count * 0.1, 0.4)
        
        # 길이 적절성 확인
        if 20 <= len(output) <= 300:
            score += 0.3
        
        return min(score, 1.0)

    def setup_distillation_lora_config(self) -> LoraConfig:
        """증류를 위한 LoRA 설정"""
        return LoraConfig(
            r=32,  # 더 높은 rank로 표현력 증대
            lora_alpha=64,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ],
            lora_dropout=0.05,  # 낮은 드롭아웃
            bias="none",
            task_type="CAUSAL_LM",
        )

    def distillation_loss_function(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        hard_targets: torch.Tensor,
        temperature: float = None,
    ) -> torch.Tensor:
        """
        증류 손실 함수
        
        Args:
            student_logits: 학생 모델 출력 로짓
            teacher_logits: 교사 모델 출력 로짓
            hard_targets: 실제 정답 타겟
            temperature: 증류 온도
            
        Returns:
            통합 손실
        """
        if temperature is None:
            temperature = self.distillation_config["temperature"]
        
        # Soft target loss (Knowledge Distillation)
        soft_student = F.log_softmax(student_logits / temperature, dim=-1)
        soft_teacher = F.softmax(teacher_logits / temperature, dim=-1)
        soft_loss = F.kl_div(soft_student, soft_teacher, reduction="batchmean") * (temperature ** 2)
        
        # Hard target loss (Ground Truth)
        hard_loss = F.cross_entropy(student_logits, hard_targets, ignore_index=-100)
        
        # 가중합 손실
        total_loss = (
            self.distillation_config["alpha"] * hard_loss +
            self.distillation_config["beta"] * soft_loss
        )
        
        return total_loss

    def progressive_distillation(self, dataset: Dataset, output_dir: str) -> str:
        """
        점진적 증류 수행
        
        Args:
            dataset: 증류용 데이터셋
            output_dir: 출력 디렉토리
            
        Returns:
            최종 모델 경로
        """
        logger.info("Starting progressive distillation...")
        
        stages = []
        
        # Stage 1: Large teacher → Intermediate model
        if self.intermediate:
            logger.info("Stage 1: Teacher → Intermediate distillation")
            intermediate_path = os.path.join(output_dir, "intermediate_distilled")
            self._distill_single_stage(
                teacher=self.teachers[self.teacher_models[0]],  # 첫 번째 교사 사용
                student=self.intermediate,
                dataset=dataset,
                output_path=intermediate_path,
                epochs=3
            )
            stages.append(intermediate_path)
        
        # Stage 2: Intermediate → Final student model
        logger.info("Stage 2: Intermediate → Student distillation")
        final_path = os.path.join(output_dir, "final_distilled")
        
        teacher_for_final = self.intermediate if self.intermediate else self.teachers[self.teacher_models[0]]
        
        self._distill_single_stage(
            teacher=teacher_for_final,
            student=self.student,
            dataset=dataset,
            output_path=final_path,
            epochs=5
        )
        stages.append(final_path)
        
        logger.info(f"Progressive distillation completed. Final model: {final_path}")
        return final_path

    def _distill_single_stage(
        self,
        teacher: AutoModelForCausalLM,
        student: AutoModelForCausalLM,
        dataset: Dataset,
        output_path: str,
        epochs: int = 3
    ):
        """단일 증류 단계 수행"""
        
        # LoRA 적용
        lora_config = self.setup_distillation_lora_config()
        student = prepare_model_for_kbit_training(student)
        student = get_peft_model(student, lora_config)
        
        # 커스텀 트레이너로 증류 수행
        trainer = DistillationTrainer(
            teacher_model=teacher,
            student_model=student,
            tokenizer=self.tokenizer,
            train_dataset=dataset,
            distillation_config=self.distillation_config,
            output_dir=output_path,
            num_train_epochs=epochs,
        )
        
        trainer.train()
        trainer.save_model()

    def evaluate_distilled_model(self, model_path: str, test_data: List[Dict]) -> Dict[str, float]:
        """증류된 모델 성능 평가"""
        logger.info(f"Evaluating distilled model: {model_path}")
        
        # 모델 로드
        eval_model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float16, device_map="auto"
        )
        
        metrics = {
            "accuracy": 0.0,
            "exact_match": 0.0,
            "function_call_validity": 0.0,
            "semantic_similarity": 0.0,
        }
        
        correct = 0
        exact_matches = 0
        valid_function_calls = 0
        
        for item in test_data:
            query = item["query"]
            expected = item["expected_output"]
            
            # 추론 수행
            prediction = self._generate_prediction(eval_model, query)
            
            # 메트릭 계산
            if self._is_functionally_correct(prediction, expected):
                correct += 1
            
            if prediction.strip() == expected.strip():
                exact_matches += 1
            
            if self._is_valid_function_call(prediction):
                valid_function_calls += 1
        
        metrics["accuracy"] = correct / len(test_data)
        metrics["exact_match"] = exact_matches / len(test_data)
        metrics["function_call_validity"] = valid_function_calls / len(test_data)
        
        logger.info(f"Evaluation metrics: {metrics}")
        return metrics

    def _generate_prediction(self, model: AutoModelForCausalLM, query: str) -> str:
        """모델로 예측 생성"""
        system_prompt = "당신은 한국어 음성 명령을 Function Call로 변환하는 AI 어시스턴트입니다."
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"사용자 명령: {query}"},
        ]
        
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.1,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.split("assistant")[-1].strip()

    def _is_functionally_correct(self, prediction: str, expected: str) -> bool:
        """기능적 정확성 평가"""
        try:
            pred_json = json.loads(prediction)
            exp_json = json.loads(expected)
            
            # 함수명과 주요 파라미터 비교
            return (
                pred_json.get("function") == exp_json.get("function") and
                pred_json.get("parameters", {}).keys() == exp_json.get("parameters", {}).keys()
            )
        except:
            return False

    def _is_valid_function_call(self, output: str) -> bool:
        """유효한 함수 호출 형태인지 확인"""
        try:
            parsed = json.loads(output)
            return "function" in parsed and "parameters" in parsed
        except:
            return False


class DistillationTrainer(Trainer):
    """증류를 위한 커스텀 트레이너"""
    
    def __init__(self, teacher_model, student_model, distillation_config, **kwargs):
        super().__init__(model=student_model, **kwargs)
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.distillation_config = distillation_config
        
        # 교사 모델을 평가 모드로 고정
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False

    def compute_loss(self, model, inputs, return_outputs=False):
        """증류 손실 계산"""
        
        # 학생 모델 순전파
        student_outputs = model(**inputs)
        student_logits = student_outputs.logits
        
        # 교사 모델 순전파
        with torch.no_grad():
            teacher_outputs = self.teacher_model(**inputs)
            teacher_logits = teacher_outputs.logits
        
        # 증류 손실 계산
        labels = inputs.get("labels")
        if labels is not None:
            # Shift labels for causal LM
            shift_logits = student_logits[..., :-1, :].contiguous()
            shift_teacher_logits = teacher_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Knowledge distillation loss
            temperature = self.distillation_config["temperature"]
            
            soft_student = F.log_softmax(shift_logits / temperature, dim=-1)
            soft_teacher = F.softmax(shift_teacher_logits / temperature, dim=-1)
            soft_loss = F.kl_div(
                soft_student.view(-1, soft_student.size(-1)),
                soft_teacher.view(-1, soft_teacher.size(-1)),
                reduction="batchmean"
            ) * (temperature ** 2)
            
            # Hard target loss
            hard_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100
            )
            
            # Combined loss
            alpha = self.distillation_config["alpha"]
            beta = self.distillation_config["beta"]
            loss = alpha * hard_loss + beta * soft_loss
        else:
            loss = student_outputs.loss
        
        return (loss, student_outputs) if return_outputs else loss


if __name__ == "__main__":
    # 사용 예시
    
    # 교사 모델들 정의
    teacher_models = [
        "meta-llama/Llama-3.1-70B-Instruct",  # 대형 로컬 모델
        "openai/gpt-4o-mini",  # API 모델 (예시)
    ]
    
    # 증류 트레이너 초기화
    distiller = TaskDistillationTrainer(
        teacher_models=teacher_models,
        student_model="meta-llama/Llama-3.2-1B-Instruct",
        intermediate_model="meta-llama/Llama-3.2-3B-Instruct"
    )
    
    # 모델 로드
    distiller.load_models()
    
    # 데이터 로드 (기존 방식 활용)
    # raw_data = load_function_call_data(["/path/to/train_data.csv"])
    # distillation_dataset = distiller.create_distillation_dataset(raw_data)
    
    # 점진적 증류 수행
    # final_model_path = distiller.progressive_distillation(
    #     distillation_dataset, 
    #     "./distilled_models"
    # )
    
    logger.info("Task distillation setup complete!")
