#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
지역 마스터 데이터 처리 클래스
시도-시군구 매핑과 면적 정보를 관리
"""

import pandas as pd
import logging
from typing import Dict, List, Tuple
from pathlib import Path


class AreaMaster:
    """지역 마스터 데이터를 관리하는 클래스"""
    
    def __init__(self, data_path: str = "data/raw/지역_면적_utf8.csv"):
        """
        Args:
            data_path (str): 지역 면적 데이터 파일 경로
        """
        self.data_path = Path(data_path)
        self.area_data = None
        self.sido_sigungu_mapping = {}
        self.logger = self._setup_logger()
        
    def _setup_logger(self) -> logging.Logger:
        """로거 설정"""
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def load_area_data(self) -> pd.DataFrame:
        """지역 면적 데이터 로드"""
        try:
            self.logger.info(f"지역 면적 데이터 로드 중: {self.data_path}")
            self.area_data = pd.read_csv(self.data_path, encoding='utf-8')
            self.logger.info(f"데이터 로드 완료: {len(self.area_data)}개 지역")
            return self.area_data
        except Exception as e:
            self.logger.error(f"데이터 로드 실패: {e}")
            raise
    
    def create_sido_sigungu_mapping(self) -> Dict[str, str]:
        """시도-시군구 매핑 딕셔너리 생성"""
        if self.area_data is None:
            self.load_area_data()
        
        self.sido_sigungu_mapping = dict(
            zip(self.area_data['시군구'], self.area_data['시도'])
        )
        
        self.logger.info(f"시도-시군구 매핑 생성 완료: {len(self.sido_sigungu_mapping)}개")
        return self.sido_sigungu_mapping
    
    def get_sido_by_sigungu(self, sigungu: str) -> str:
        """시군구명으로 시도명 조회"""
        if not self.sido_sigungu_mapping:
            self.create_sido_sigungu_mapping()
        
        return self.sido_sigungu_mapping.get(sigungu, "알수없음")
    
    def get_area_by_sigungu(self, sigungu: str) -> float:
        """시군구명으로 면적 조회"""
        if self.area_data is None:
            self.load_area_data()
        
        area_row = self.area_data[self.area_data['시군구'] == sigungu]
        if not area_row.empty:
            return area_row.iloc[0]['면적']
        return 0.0
    
    def get_all_regions(self) -> List[Tuple[str, str, float]]:
        """모든 지역 정보 반환 (시도, 시군구, 면적)"""
        if self.area_data is None:
            self.load_area_data()
        
        regions = []
        for _, row in self.area_data.iterrows():
            regions.append((
                row['시도'],
                row['시군구'],
                row['면적']
            ))
        
        return regions
    
    def validate_region_data(self) -> bool:
        """지역 데이터 유효성 검사"""
        if self.area_data is None:
            self.load_area_data()
        
        # 필수 칼럼 확인
        required_columns = ['시도', '시군구', '면적']
        missing_columns = [col for col in required_columns if col not in self.area_data.columns]
        
        if missing_columns:
            self.logger.error(f"필수 칼럼 누락: {missing_columns}")
            return False
        
        # 중복 시군구 확인
        duplicates = self.area_data['시군구'].duplicated()
        if duplicates.any():
            duplicate_sigungus = self.area_data[duplicates]['시군구'].tolist()
            self.logger.warning(f"중복 시군구 발견: {duplicate_sigungus}")
        
        # 면적 데이터 타입 확인
        try:
            self.area_data['면적'] = pd.to_numeric(self.area_data['면적'], errors='coerce')
        except Exception as e:
            self.logger.error(f"면적 데이터 변환 실패: {e}")
            return False
        
        self.logger.info("지역 데이터 유효성 검사 완료")
        return True
    
    def get_summary(self) -> Dict:
        """지역 데이터 요약 정보 반환"""
        if self.area_data is None:
            self.load_area_data()
        
        summary = {
            '총_지역_수': len(self.area_data),
            '시도_수': self.area_data['시도'].nunique(),
            '시군구_수': self.area_data['시군구'].nunique(),
            '면적_총합': self.area_data['면적'].sum(),
            '면적_평균': self.area_data['면적'].mean(),
            '시도별_지역수': self.area_data['시도'].value_counts().to_dict()
        }
        
        return summary
    
    def export_mapping(self, output_path: str = "data/processed/시도_시군구_매핑.csv"):
        """시도-시군구 매핑을 CSV로 내보내기"""
        if self.area_data is None:
            self.load_area_data()
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        mapping_df = self.area_data[['시도', '시군구', '면적']].copy()
        mapping_df.to_csv(output_path, index=False, encoding='utf-8')
        
        self.logger.info(f"매핑 데이터 내보내기 완료: {output_path}")
        return output_path


if __name__ == "__main__":
    # 테스트 코드
    area_master = AreaMaster()
    
    # 데이터 로드 및 검증
    area_master.load_area_data()
    is_valid = area_master.validate_region_data()
    
    if is_valid:
        # 요약 정보 출력
        summary = area_master.get_summary()
        print("=== 지역 데이터 요약 ===")
        for key, value in summary.items():
            print(f"{key}: {value}")
        
        # 매핑 생성
        area_master.create_sido_sigungu_mapping()
        
        # 예시 조회
        print(f"\n=== 예시 조회 ===")
        print(f"종로구의 시도: {area_master.get_sido_by_sigungu('종로구')}")
        print(f"종로구의 면적: {area_master.get_area_by_sigungu('종로구')}")
        
        # 매핑 내보내기
        area_master.export_mapping()
    else:
        print("데이터 유효성 검사 실패")
