#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
데이터 전처리 메인 실행 파일
전체 전처리 파이프라인을 실행하여 최종 통합 데이터 생성
"""

import logging
import sys
from pathlib import Path
from .data_merger import DataMerger


def setup_logging():
    """로깅 설정"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/preprocess_pipeline.log', encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )


def main():
    """메인 실행 함수"""
    # 로깅 설정
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("=== 데이터 전처리 파이프라인 시작 ===")
        
        # 데이터 병합기 생성
        data_merger = DataMerger()
        
        # 전체 파이프라인 실행 (2019~2023년)
        logger.info("2019~2023년 데이터 전처리 파이프라인 실행")
        final_data = data_merger.run_complete_pipeline(start_year=2019, end_year=2023)
        
        # 최종 결과 요약
        logger.info("=== 전처리 완료 ===")
        logger.info(f"최종 데이터 행 수: {len(final_data)}")
        logger.info(f"최종 데이터 칼럼: {list(final_data.columns)}")
        
        # 데이터 미리보기
        logger.info("=== 데이터 미리보기 ===")
        logger.info(f"\n{final_data.head()}")
        
        # 요약 통계
        summary = data_merger.get_final_summary()
        logger.info("=== 최종 데이터 요약 ===")
        for key, value in summary.items():
            logger.info(f"{key}: {value}")
        
        logger.info("=== 데이터 전처리 파이프라인 완료 ===")
        
        return True
        
    except Exception as e:
        logger.error(f"파이프라인 실행 중 오류 발생: {e}")
        return False


if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ 데이터 전처리 파이프라인이 성공적으로 완료되었습니다!")
        print("📁 결과 파일: data/processed/최종_통합데이터.csv")
    else:
        print("\n❌ 데이터 전처리 파이프라인 실행 중 오류가 발생했습니다.")
        print("📋 로그 파일을 확인해주세요: logs/preprocess_pipeline.log")
        sys.exit(1)
