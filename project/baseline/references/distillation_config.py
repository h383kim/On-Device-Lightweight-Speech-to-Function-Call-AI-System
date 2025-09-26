"""
Task Distillation Configuration
Centralized configuration management for distillation experiments
"""

import json
import os
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional


@dataclass
class TeacherModelConfig:
    """교사 모델 설정"""
    name: str
    type: str  # "local", "openai", "anthropic", "huggingface"
    api_key_env: Optional[str] = None
    model_params: Dict[str, Any] = None
    weight: float = 1.0  # 앙상블에서의 가중치
    
    def __post_init__(self):
        if self.model_params is None:
            self.model_params = {}


@dataclass
class DistillationConfig:
    """증류 설정"""
    # 온도 설정
    temperature: float = 4.0
    adaptive_temperature: bool = True
    temperature_schedule: str = "cosine"  # "constant", "linear", "cosine"
    
    # 손실 가중치
    alpha: float = 0.7  # Hard target weight
    beta: float = 0.3   # Soft target weight
    attention_weight: float = 0.1
    hidden_weight: float = 0.1
    
    # 점진적 증류
    progressive: bool = True
    num_stages: int = 2
    
    # 앙상블 설정
    ensemble_method: str = "weighted_average"  # "voting", "weighted_average", "quality_based"
    
    # 품질 평가
    quality_threshold: float = 0.7
    use_quality_filtering: bool = True


@dataclass
class TrainingConfig:
    """훈련 설정"""
    # 기본 설정
    num_epochs: int = 5
    batch_size: int = 4
    learning_rate: float = 2e-4
    warmup_steps: int = 100
    
    # LoRA 설정
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    target_modules: List[str] = None
    
    # 최적화
    optimizer: str = "adamw"
    weight_decay: float = 0.01
    gradient_clipping: float = 1.0
    
    # 스케줄링
    scheduler: str = "cosine"  # "linear", "cosine", "polynomial"
    
    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]


@dataclass
class EvaluationConfig:
    """평가 설정"""
    metrics: List[str] = None
    eval_steps: int = 500
    save_best: bool = True
    early_stopping_patience: int = 3
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = [
                "exact_match",
                "function_call_validity",
                "semantic_similarity",
                "rouge_l",
                "bleu"
            ]


@dataclass
class TaskDistillationExperimentConfig:
    """전체 실험 설정"""
    # 실험 메타데이터
    experiment_name: str
    description: str
    version: str = "1.0"
    
    # 모델 설정
    student_model: str = "meta-llama/Llama-3.2-1B-Instruct"
    intermediate_model: Optional[str] = "meta-llama/Llama-3.2-3B-Instruct"
    teacher_models: List[TeacherModelConfig] = None
    
    # 설정 구성요소
    distillation: DistillationConfig = None
    training: TrainingConfig = None
    evaluation: EvaluationConfig = None
    
    # 경로 설정
    data_dir: str = "/mnt/elice/dataset"
    output_dir: str = "./distillation_experiments"
    cache_dir: str = "./cache"
    log_dir: str = "./logs"
    
    # 실행 제어
    use_wandb: bool = True
    wandb_project: str = "korean-function-call-distillation"
    save_checkpoints: bool = True
    checkpoint_steps: int = 1000
    
    def __post_init__(self):
        if self.teacher_models is None:
            self.teacher_models = [
                TeacherModelConfig(
                    name="meta-llama/Llama-3.1-8B-Instruct",
                    type="local",
                    weight=1.0
                )
            ]
        
        if self.distillation is None:
            self.distillation = DistillationConfig()
        
        if self.training is None:
            self.training = TrainingConfig()
        
        if self.evaluation is None:
            self.evaluation = EvaluationConfig()
    
    def save_config(self, path: str):
        """설정을 JSON 파일로 저장"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(asdict(self), f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load_config(cls, path: str):
        """JSON 파일에서 설정 로드"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 중첩된 dataclass 객체 복원
        if 'teacher_models' in data:
            data['teacher_models'] = [
                TeacherModelConfig(**tm) for tm in data['teacher_models']
            ]
        
        if 'distillation' in data:
            data['distillation'] = DistillationConfig(**data['distillation'])
        
        if 'training' in data:
            data['training'] = TrainingConfig(**data['training'])
        
        if 'evaluation' in data:
            data['evaluation'] = EvaluationConfig(**data['evaluation'])
        
        return cls(**data)


# 사전 정의된 실험 설정들
EXPERIMENT_CONFIGS = {
    "baseline_distillation": TaskDistillationExperimentConfig(
        experiment_name="baseline_distillation",
        description="기본적인 teacher-student 증류 실험",
        teacher_models=[
            TeacherModelConfig(
                name="meta-llama/Llama-3.1-8B-Instruct",
                type="local",
                weight=1.0
            )
        ],
        distillation=DistillationConfig(
            temperature=4.0,
            alpha=0.7,
            beta=0.3,
            progressive=False
        )
    ),
    
    "progressive_distillation": TaskDistillationExperimentConfig(
        experiment_name="progressive_distillation",
        description="점진적 증류 실험 (Large → Medium → Small)",
        intermediate_model="meta-llama/Llama-3.2-3B-Instruct",
        teacher_models=[
            TeacherModelConfig(
                name="meta-llama/Llama-3.1-70B-Instruct",
                type="local",
                weight=1.0
            )
        ],
        distillation=DistillationConfig(
            progressive=True,
            num_stages=2,
            temperature=3.0
        )
    ),
    
    "ensemble_distillation": TaskDistillationExperimentConfig(
        experiment_name="ensemble_distillation",
        description="다중 교사 앙상블 증류",
        teacher_models=[
            TeacherModelConfig(
                name="meta-llama/Llama-3.1-8B-Instruct",
                type="local",
                weight=0.4
            ),
            TeacherModelConfig(
                name="microsoft/DialoGPT-large",
                type="local",
                weight=0.3
            ),
            TeacherModelConfig(
                name="openai/gpt-4o-mini",
                type="openai",
                api_key_env="OPENAI_API_KEY",
                weight=0.3
            )
        ],
        distillation=DistillationConfig(
            ensemble_method="weighted_average",
            use_quality_filtering=True,
            quality_threshold=0.8
        )
    ),
    
    "attention_distillation": TaskDistillationExperimentConfig(
        experiment_name="attention_distillation",
        description="어텐션 패턴 증류 포함",
        teacher_models=[
            TeacherModelConfig(
                name="meta-llama/Llama-3.1-8B-Instruct",
                type="local",
                weight=1.0
            )
        ],
        distillation=DistillationConfig(
            attention_weight=0.2,
            hidden_weight=0.1,
            temperature=3.5
        )
    )
}


def get_experiment_config(name: str) -> TaskDistillationExperimentConfig:
    """사전 정의된 실험 설정 반환"""
    if name not in EXPERIMENT_CONFIGS:
        raise ValueError(f"Unknown experiment config: {name}. Available: {list(EXPERIMENT_CONFIGS.keys())}")
    return EXPERIMENT_CONFIGS[name]


def create_custom_config(
    experiment_name: str,
    student_model: str = "meta-llama/Llama-3.2-1B-Instruct",
    teacher_models: List[str] = None,
    **kwargs
) -> TaskDistillationExperimentConfig:
    """커스텀 실험 설정 생성"""
    
    if teacher_models is None:
        teacher_models = ["meta-llama/Llama-3.1-8B-Instruct"]
    
    teacher_configs = [
        TeacherModelConfig(name=tm, type="local", weight=1.0/len(teacher_models))
        for tm in teacher_models
    ]
    
    config = TaskDistillationExperimentConfig(
        experiment_name=experiment_name,
        description=f"Custom experiment: {experiment_name}",
        student_model=student_model,
        teacher_models=teacher_configs,
        **kwargs
    )
    
    return config


if __name__ == "__main__":
    # 설정 파일 생성 예시
    
    # 기본 실험 설정 저장
    for name, config in EXPERIMENT_CONFIGS.items():
        config_path = f"./configs/{name}.json"
        config.save_config(config_path)
        print(f"Saved config: {config_path}")
    
    # 커스텀 설정 생성 및 저장
    custom_config = create_custom_config(
        experiment_name="custom_korean_distillation",
        student_model="meta-llama/Llama-3.2-1B-Instruct",
        teacher_models=[
            "meta-llama/Llama-3.1-8B-Instruct",
            "meta-llama/Llama-3.1-70B-Instruct"
        ]
    )
    
    custom_config.save_config("./configs/custom_korean_distillation.json")
    print("Custom config saved!")
    
    # 설정 로드 테스트
    loaded_config = TaskDistillationExperimentConfig.load_config("./configs/custom_korean_distillation.json")
    print(f"Loaded experiment: {loaded_config.experiment_name}")
