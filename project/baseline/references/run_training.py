import argparse
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# 현재 디렉토리를 Python 경로에 추가
sys.path.append(str(Path(__file__).parent))

from llama_finetune import LlamaFunctionCallTrainer
from peft_merger import merge_peft_model


def setup_logger(log_file: str = "run_training.log"):
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

    return logger


def parse_args():
    """명령행 인자를 파싱합니다."""
    parser = argparse.ArgumentParser(description="Llama 3.2 Function Call Fine-tuning")

    parser.add_argument(
        "--model_name", type=str, default="meta-llama/Llama-3.2-1B-Instruct", help="사용할 Llama 모델명"
    )
    parser.add_argument("--data_dir", type=str, default="/mnt/elice/dataset", help="Function Call 데이터 디렉토리")
    parser.add_argument(
        "--output_dir", type=str, default="./llama_function_call_model", help="Fine-tuned 모델 저장 경로"
    )
    parser.add_argument("--merged_dir", type=str, default="./llama_function_call_merged", help="병합된 모델 저장 경로")
    parser.add_argument("--num_train_epochs", type=int, default=5, help="학습 에포크 수")
    parser.add_argument("--skip_training", action="store_true", help="학습을 건너뛰고 병합부터 수행")
    parser.add_argument("--batch_size", type=int, default=4, help="배치 크기")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="학습률")
    parser.add_argument("--log_file", type=str, default="run_training.log", help="로깅 파일 경로")

    return parser.parse_args()


def get_data_files(data_dir: str) -> list:
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


def main():
    """메인 실행 함수"""
    args = parse_args()
    args_dict = vars(args)

    # 로깅 파일 경로를 인자로 받아서 새로 설정
    global logger
    logger = setup_logger(args.log_file)

    logger.info("=== Llama 3.2 Function Call Fine-tuning 시작 ===")
    logger.info(f"모델: {args.model_name}")
    logger.info(f"데이터 디렉토리: {args.data_dir}")
    logger.info(f"출력 디렉토리: {args.output_dir}")

    # 학습 단계
    logger.info("=== 1단계: 모델 Fine-tuning ===")

    # 트레이너 초기화
    trainer = LlamaFunctionCallTrainer(args.model_name)
    if args.skip_training:
        logger.info("학습을 건너뛰고 병합부터 수행합니다.")
    else:
        # 모델과 토크나이저 로드
        trainer.load_model_and_tokenizer()

        # 데이터 로드
        data_paths = get_data_files(args.data_dir)
        if not data_paths:
            logger.error("사용할 수 있는 데이터 파일이 없습니다.")
            return

        raw_data = trainer.load_function_call_data(data_paths)

        # 학습 데이터 준비
        dataset = trainer.prepare_training_data(raw_data)

        # Fine-tuning 수행
        trainer.fine_tune_model(dataset, args.output_dir, args_dict)

        logger.info("Fine-tuning 완료!")

    if not os.path.exists(args.output_dir):
        logger.error(f"Fine-tuned 모델이 없습니다: {args.output_dir}")
        return

    logger.info("=== 2단계: 모델 병합 ===")
    merge_peft_model(args.model_name, args.output_dir, args.merged_dir)

    logger.info("=== 전체 파이프라인 완료! ===")


if __name__ == "__main__":
    load_dotenv(override=True)
    main()
