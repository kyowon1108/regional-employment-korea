#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
최종 데이터 생성 메인 실행 파일
"""

import logging
from pathlib import Path
from .data_processor import DataProcessor
from .config import Config

def setup_logging():
    """로깅 설정"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / "final_data_generation.log", encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

def main():
    """메인 실행 함수"""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("최종 데이터 생성 시작")
        
        # 설정 로드
        config = Config()
        
        # 데이터 프로세서 생성 및 실행
        processor = DataProcessor(config)
        processor.process_all_data()
        
        logger.info("최종 데이터 생성 완료")
        
    except Exception as e:
        logger.error(f"데이터 생성 중 오류 발생: {e}")
        raise

if __name__ == "__main__":
    main()
