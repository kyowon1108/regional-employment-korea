#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
지역 정보 처리 클래스
"""

import pandas as pd
import logging
from pathlib import Path
from typing import Dict, Any

class AreaProcessor:
    """지역 정보 처리 클래스"""
    
    def __init__(self, config):
        """초기화"""
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.area_data = None
    
    def load_area_data(self) -> pd.DataFrame:
        """지역 면적 데이터 로드"""
        try:
            self.logger.info("지역 면적 데이터 로드 시작")
            
            # UTF-8로 파일 읽기
            df = pd.read_csv(self.config.area_file, encoding='utf-8')
            
            # 칼럼명 정리
            df.columns = ['시도', '시군구', '면적', '제조업비중', '서비스업비중']
            
            # 면적을 숫자로 변환
            df['면적'] = pd.to_numeric(df['면적'], errors='coerce')
            
            # 제조업비중, 서비스업비중을 숫자로 변환
            df['제조업비중'] = pd.to_numeric(df['제조업비중'], errors='coerce')
            df['서비스업비중'] = pd.to_numeric(df['서비스업비중'], errors='coerce')
            
            # 결측값 처리
            df = df.dropna(subset=['시도', '시군구', '면적'])
            
            self.logger.info(f"지역 면적 데이터 로드 완료: {len(df)}개 지역")
            self.area_data = df
            
            return df
            
        except Exception as e:
            self.logger.error(f"지역 면적 데이터 로드 실패: {e}")
            raise
    
    def get_area_data(self) -> pd.DataFrame:
        """지역 데이터 반환"""
        if self.area_data is None:
            return self.load_area_data()
        return self.area_data
    
    def get_unique_regions(self) -> pd.DataFrame:
        """고유 지역 정보 반환 (시도, 시군구, 면적만)"""
        df = self.get_area_data()
        return df[['시도', '시군구', '면적']].copy()
    
    def get_industry_ratios(self) -> pd.DataFrame:
        """산업별 비중 정보 반환"""
        df = self.get_area_data()
        return df[['시도', '시군구', '제조업비중', '서비스업비중']].copy()
