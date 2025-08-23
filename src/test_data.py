#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
데이터 처리 테스트 스크립트
"""

import logging
from pathlib import Path
from .config import Config
from .data_processor import DataProcessor

def test_data_processing():
    """데이터 처리 테스트"""
    # 로깅 설정
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("데이터 처리 테스트 시작")
        
        # 설정 로드
        config = Config()
        logger.info("설정 로드 완료")
        
        # 데이터 프로세서 생성
        processor = DataProcessor(config)
        logger.info("데이터 프로세서 생성 완료")
        
        # 데이터 처리 실행
        processor.process_all_data()
        logger.info("데이터 처리 실행 완료")
        
        # 결과 확인
        final_data = processor.get_final_data()
        summary = processor.get_summary_stats()
        
        logger.info("=== 최종 데이터 요약 ===")
        logger.info(f"총 레코드 수: {summary['총_레코드_수']}")
        logger.info(f"지역 수: {summary['지역_수']}")
        logger.info(f"연도 범위: {summary['연도_범위']}")
        logger.info(f"반기: {summary['반기']}")
        logger.info(f"시도 목록: {summary['시도_목록']}")
        
        logger.info("=== 최종 데이터 샘플 ===")
        logger.info(f"데이터 형태: {final_data.shape}")
        logger.info(f"칼럼: {list(final_data.columns)}")
        logger.info("\n처음 5개 행:")
        logger.info(final_data.head().to_string())
        
        logger.info("테스트 완료!")
        
    except Exception as e:
        logger.error(f"테스트 중 오류 발생: {e}")
        raise

if __name__ == "__main__":
    test_data_processing()
