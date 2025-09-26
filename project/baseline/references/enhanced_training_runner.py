#!/usr/bin/env python3
"""
Enhanced Training Runner with Task Distillation Support
Extends the baseline training with advanced distillation capabilities
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv

# 현재 디렉토리를 Python 경로에 추가
sys.path.append(str(Path(__file__).parent))

from llama_finetune import LlamaFunctionCallTrainer
from peft_merger import merge_peft_model

# Task distillation import (if available)
try:
    from task_distillation_trainer import TaskDistillationTrainer
    DISTILLATION_AVAILABLE = True
except ImportError:
    DISTILLATION_AVAILABLE = False
    logging.warning("Task distillation trainer not available. Falling back to standard fine-tuning.")


def setup_logger(log_file: str = "enhanced_training.log"):
    """로깅을 파일에 기록하도록 설정합니다."""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # 포맷 정의
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # 파일 핸들러
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # 콘솔 핸들러
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger


def parse_args():
    """명령행 인자를 파싱합니다."""
    parser = argparse.ArgumentParser(description="Enhanced Llama 3.2 Function Call Training with Task Distillation")

    # 기본 모델 설정
    parser.add_argument(
        "--model_name", type=str, default="meta-llama/Llama-3.2-1B-Instruct", help="사용할 학생 모델명"
    )
    parser.add_argument("--data_dir", type=str, default="/mnt/elice/dataset", help="Function Call 데이터 디렉토리")
    parser.add_argument(
        "--output_dir", type=str, default="./enhanced_function_call_model", help="Fine-tuned 모델 저장 경로"
    )
    parser.add_argument("--merged_dir", type=str, default="./enhanced_function_call_merged", help="병합된 모델 저장 경로")

    # 훈련 설정
    parser.add_argument("--num_train_epochs", type=int, default=5, help="학습 에포크 수")
    parser.add_argument("--batch_size", type=int, default=4, help="배치 크기")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="학습률")
    parser.add_argument("--log_file", type=str, default="enhanced_training.log", help="로깅 파일 경로")

    # Task Distillation 설정
    parser.add_argument("--use_distillation", action="store_true", help="Task distillation 사용")
    parser.add_argument(
        "--teacher_models",
        nargs="+",
        default=["meta-llama/Llama-3.1-8B-Instruct"],
        help="교사 모델 리스트"
    )
    parser.add_argument(
        "--intermediate_model",
        type=str,
        default="meta-llama/Llama-3.2-3B-Instruct",
        help="중간 크기 모델 (점진적 증류용)"
    )
    parser.add_argument("--distillation_temperature", type=float, default=4.0, help="증류 온도")
    parser.add_argument("--distillation_alpha", type=float, default=0.7, help="Hard target 가중치")
    parser.add_argument("--distillation_beta", type=float, default=0.3, help="Soft target 가중치")
    
    # 고급 설정
    parser.add_argument("--progressive_distillation", action="store_true", help="점진적 증류 사용")
    parser.add_argument("--ensemble_teachers", action="store_true", help="다중 교사 앙상블 사용")
    parser.add_argument("--attention_distillation", action="store_true", help="어텐션 증류 추가")
    
    # 실행 제어
    parser.add_argument("--skip_training", action="store_true", help="학습을 건너뛰고 병합부터 수행")
    parser.add_argument("--evaluation_only", action="store_true", help="평가만 수행")

    return parser.parse_args()


def get_data_files(data_dir: str) -> List[str]:
    """데이터 파일 경로를 가져옵니다."""
    data_files = ["train_data.csv"]

    data_paths = []
    for file in data_files:
        path = os.path.join(data_dir, file)
        if os.path.exists(path):
            data_paths.append(path)
            logger.info(f"Found data file: {path}")
        else:
            logger.warning(f"Data file not found: {path}")

    return data_paths


def run_standard_fine_tuning(args, trainer: LlamaFunctionCallTrainer, data_paths: List[str]):
    """표준 fine-tuning 실행"""
    logger.info("=== 표준 LoRA Fine-tuning 실행 ===")
    
    # 모델과 토크나이저 로드
    trainer.load_model_and_tokenizer()

    # 데이터 로드 및 준비
    raw_data = trainer.load_function_call_data(data_paths)
    dataset = trainer.prepare_training_data(raw_data)

    # Fine-tuning 수행
    args_dict = vars(args)
    trainer.fine_tune_model(dataset, args.output_dir, args_dict)

    logger.info("표준 Fine-tuning 완료!")
    return args.output_dir


def run_task_distillation(args, data_paths: List[str]):
    """Task distillation 실행"""
    logger.info("=== Task Distillation 실행 ===")
    
    # Distillation trainer 초기화
    distillation_config = {
        "temperature": args.distillation_temperature,
        "alpha": args.distillation_alpha,
        "beta": args.distillation_beta,
        "attention_weight": 0.1 if args.attention_distillation else 0.0,
        "hidden_weight": 0.1 if args.attention_distillation else 0.0,
    }
    
    distiller = TaskDistillationTrainer(
        teacher_models=args.teacher_models,
        student_model=args.model_name,
        intermediate_model=args.intermediate_model if args.progressive_distillation else None,
    )
    
    # 증류 설정 업데이트
    distiller.distillation_config.update(distillation_config)
    
    # 모델 로드
    distiller.load_models()
    
    # 데이터 로드 및 증류용 데이터셋 생성
    import pandas as pd
    raw_data_list = []
    for data_path in data_paths:
        if os.path.exists(data_path):
            df = pd.read_csv(data_path, encoding="utf-8", index_col=0)
            raw_data_list.append(df)
    
    if not raw_data_list:
        raise ValueError("No training data found!")
    
    raw_data = pd.concat(raw_data_list)
    
    # 증류용 데이터셋 생성 (교사 모델 출력 포함)
    distillation_dataset = distiller.create_distillation_dataset(raw_data)
    
    # 증류 수행
    if args.progressive_distillation:
        final_model_path = distiller.progressive_distillation(
            distillation_dataset, 
            args.output_dir
        )
    else:
        # 단일 단계 증류
        distiller._distill_single_stage(
            teacher=distiller.teachers[args.teacher_models[0]],
            student=distiller.student,
            dataset=distillation_dataset,
            output_path=args.output_dir,
            epochs=args.num_train_epochs
        )
        final_model_path = args.output_dir
    
    logger.info(f"Task Distillation 완료! 최종 모델: {final_model_path}")
    return final_model_path


def evaluate_model_performance(model_path: str, test_data_path: str = None):
    """모델 성능 평가"""
    logger.info(f"=== 모델 성능 평가: {model_path} ===")
    
    # 테스트 데이터가 없으면 기본 쿼리로 테스트
    if not test_data_path or not os.path.exists(test_data_path):
        test_queries = [
            "음악을 틀어줘",
            "날씨가 어때?",
            "알람을 7시에 맞춰줘",
            "전화를 걸어줘",
            "메시지 보내줘"
        ]
        
        # 간단한 추론 테스트
        from llama_finetune import LlamaFunctionCallTrainer
        evaluator = LlamaFunctionCallTrainer()
        evaluator.test_model_inference(test_queries, model_path)
    else:
        # 정량적 평가 수행
        logger.info("정량적 평가 기능은 추후 구현 예정")


def main():
    """메인 실행 함수"""
    args = parse_args()
    
    # 로깅 설정
    global logger
    logger = setup_logger(args.log_file)
    
    logger.info("=== Enhanced Llama 3.2 Function Call Training 시작 ===")
    logger.info(f"학생 모델: {args.model_name}")
    logger.info(f"데이터 디렉토리: {args.data_dir}")
    logger.info(f"출력 디렉토리: {args.output_dir}")
    logger.info(f"증류 사용: {args.use_distillation}")
    
    if args.use_distillation:
        logger.info(f"교사 모델들: {args.teacher_models}")
        logger.info(f"점진적 증류: {args.progressive_distillation}")
        logger.info(f"앙상블 교사: {args.ensemble_teachers}")
    
    # 데이터 파일 확인
    data_paths = get_data_files(args.data_dir)
    if not data_paths:
        logger.error("사용할 수 있는 데이터 파일이 없습니다.")
        return
    
    # 평가만 수행하는 경우
    if args.evaluation_only:
        evaluate_model_performance(args.output_dir)
        return
    
    # 훈련 수행
    model_path = None
    
    if args.skip_training:
        logger.info("학습을 건너뛰고 병합부터 수행합니다.")
        model_path = args.output_dir
    elif args.use_distillation and DISTILLATION_AVAILABLE:
        # Task Distillation 실행
        try:
            model_path = run_task_distillation(args, data_paths)
        except Exception as e:
            logger.error(f"Task distillation failed: {e}")
            logger.info("Falling back to standard fine-tuning...")
            trainer = LlamaFunctionCallTrainer(args.model_name)
            model_path = run_standard_fine_tuning(args, trainer, data_paths)
    else:
        # 표준 Fine-tuning 실행
        if args.use_distillation and not DISTILLATION_AVAILABLE:
            logger.warning("Task distillation requested but not available. Using standard fine-tuning.")
        
        trainer = LlamaFunctionCallTrainer(args.model_name)
        model_path = run_standard_fine_tuning(args, trainer, data_paths)
    
    # 모델 존재 확인
    if not os.path.exists(model_path):
        logger.error(f"훈련된 모델이 없습니다: {model_path}")
        return
    
    # 모델 병합 단계
    logger.info("=== 모델 병합 단계 ===")
    try:
        merge_peft_model(args.model_name, model_path, args.merged_dir)
        logger.info(f"모델 병합 완료: {args.merged_dir}")
    except Exception as e:
        logger.error(f"모델 병합 실패: {e}")
    
    # 성능 평가
    logger.info("=== 성능 평가 ===")
    evaluate_model_performance(args.merged_dir)
    
    logger.info("=== 전체 파이프라인 완료! ===")
    
    # 최종 결과 요약
    logger.info("=" * 50)
    logger.info("최종 결과 요약:")
    logger.info(f"- 훈련 방법: {'Task Distillation' if args.use_distillation and DISTILLATION_AVAILABLE else 'Standard LoRA Fine-tuning'}")
    logger.info(f"- PEFT 모델: {model_path}")
    logger.info(f"- 병합된 모델: {args.merged_dir}")
    logger.info(f"- 로그 파일: {args.log_file}")
    logger.info("=" * 50)


if __name__ == "__main__":
    load_dotenv(override=True)
    main()
