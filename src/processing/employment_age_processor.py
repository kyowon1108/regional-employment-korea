#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
시군구 연령별 취업자 및 고용률 데이터 처리 클래스
"""

import pandas as pd
import logging
from pathlib import Path
from typing import Dict, Any

class EmploymentAgeProcessor:
    """시군구 연령별 취업자 및 고용률 데이터 처리 클래스"""
    
    def __init__(self, config):
        """초기화"""
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.employment_age_data = None
        self.age_file = Path("data/raw/시군구 연령별 취업자 및 고용률 (2021~2024반기).csv")
    
    def load_employment_age_data(self) -> pd.DataFrame:
        """시군구 연령별 취업자 및 고용률 데이터 로드"""
        try:
            self.logger.info("시군구 연령별 취업자 및 고용률 데이터 로드 시작")
            
            # CP949 인코딩으로 파일 읽기
            df = pd.read_csv(self.age_file, encoding='cp949')
            
            # 원본 칼럼명 유지 (첫 번째 행은 헤더 정보이므로 제거)
            df = df.iloc[1:].reset_index(drop=True)
            
            # 칼럼명을 의미있게 변경
            df.columns = ['행정구역별', '연령대', '2021.1.취업자', '2021.1.고용률', '2021.2.취업자', '2021.2.고용률', 
                         '2022.1.취업자', '2022.1.고용률', '2022.2.취업자', '2022.2.고용률',
                         '2023.1.취업자', '2023.1.고용률', '2023.2.취업자', '2023.2.고용률',
                         '2024.1.취업자', '2024.1.고용률', '2024.2.취업자', '2024.2.고용률']
            
            # 15-64세 연령대만 선택
            df = df[df['연령대'] == '15 - 64세'].copy()
            
            # 행정구역별을 시도와 시군구로 분리
            df[['시도', '시군구']] = df['행정구역별'].str.extract(r'(.+?)\s+(.+)')
            
            # 시도, 시군구 정리
            df['시도'] = df['시도'].str.strip()
            df['시군구'] = df['시군구'].str.strip()
            
            # 시도명 매핑 (연령별 파일의 짧은 명칭을 표준 명칭으로 변환)
            sido_mapping = {
                '서울': '서울특별시',
                '부산': '부산광역시', 
                '대구': '대구광역시',
                '인천': '인천광역시',
                '광주': '광주광역시',
                '대전': '대전광역시',
                '울산': '울산광역시',
                '세종': '세종특별자치시',
                '경기': '경기도',
                '강원': '강원특별자치도',
                '충북': '충청북도',
                '충남': '충청남도',
                '전북': '전북특별자치도',
                '전남': '전라남도',
                '경북': '경상북도',
                '경남': '경상남도',
                '제주': '제주특별자치도'
            }
            
            df['시도'] = df['시도'].map(sido_mapping).fillna(df['시도'])
            
            # 데이터를 long format으로 변환
            melted_data = []
            
            for year in [2021, 2022, 2023, 2024]:
                for half in [1, 2]:
                    취업자_col = f"{year}.{half}.취업자"
                    고용률_col = f"{year}.{half}.고용률"
                    
                    if 취업자_col in df.columns and 고용률_col in df.columns:
                        temp_df = df[['시도', '시군구', 취업자_col, 고용률_col]].copy()
                        temp_df['년도'] = year
                        temp_df['반기'] = half
                        temp_df['취업자'] = pd.to_numeric(temp_df[취업자_col], errors='coerce')
                        temp_df['고용률'] = pd.to_numeric(temp_df[고용률_col], errors='coerce')
                        temp_df = temp_df[['시도', '시군구', '년도', '반기', '취업자', '고용률']]
                        melted_data.append(temp_df)
            
            # 모든 데이터 합치기
            if melted_data:
                final_df = pd.concat(melted_data, ignore_index=True)
                
                # 결측값 처리
                final_df = final_df.dropna(subset=['시도', '시군구'])
                final_df['취업자'] = final_df['취업자'].fillna(0)
                final_df['고용률'] = final_df['고용률'].fillna(0)
                
                self.logger.info(f"시군구 연령별 취업자 및 고용률 데이터 로드 완료: {len(final_df)}개 레코드")
                self.employment_age_data = final_df
                
                return final_df
            else:
                self.logger.warning("시군구 연령별 취업자 및 고용률 데이터가 비어있음")
                return pd.DataFrame()
            
        except Exception as e:
            self.logger.error(f"시군구 연령별 취업자 및 고용률 데이터 로드 실패: {e}")
            raise
    
    def get_employment_age_data(self) -> pd.DataFrame:
        """시군구 연령별 취업자 및 고용률 데이터 반환"""
        if self.employment_age_data is None:
            return self.load_employment_age_data()
        return self.employment_age_data
    
    def get_employment_by_region_year(self) -> pd.DataFrame:
        """지역별, 연도별 취업자 및 고용률 데이터 반환"""
        df = self.get_employment_age_data()
        return df[['시도', '시군구', '년도', '반기', '취업자', '고용률']].copy()
