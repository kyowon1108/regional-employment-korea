#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
데이터 통합 및 최종 처리 클래스
"""

import pandas as pd
import logging
from pathlib import Path
from typing import Dict, Any

from .area_processor import AreaProcessor
from .e9_processor import E9Processor
from .industry_processor import IndustryProcessor
from .employment_age_processor import EmploymentAgeProcessor

class DataProcessor:
    """데이터 통합 및 최종 처리 클래스"""
    
    def __init__(self, config):
        """초기화"""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 각 데이터 프로세서 초기화
        self.area_processor = AreaProcessor(config)
        self.e9_processor = E9Processor(config)
        self.industry_processor = IndustryProcessor(config)
        self.employment_age_processor = EmploymentAgeProcessor(config)
        
        self.final_data = None
    
    def process_all_data(self):
        """모든 데이터 처리 및 통합"""
        try:
            self.logger.info("전체 데이터 처리 시작")
            
            # 1. 기본 지역 정보 로드
            area_data = self.area_processor.get_unique_regions()
            self.logger.info(f"지역 데이터 로드 완료: {len(area_data)}개 지역")
            
            # 2. E9 체류자 데이터 로드
            e9_data = self.e9_processor.get_e9_by_region_year()
            self.logger.info(f"E9 데이터 로드 완료: {len(e9_data)}개 레코드")
            
            # 3. 산업별 종사자 데이터 로드
            industry_data = self.industry_processor.get_industry_by_region_year()
            self.logger.info(f"산업별 데이터 로드 완료: {len(industry_data)}개 레코드")
            
            # 4. 시군구 연령별 취업자 및 고용률 데이터 로드
            employment_data = self.employment_age_processor.get_employment_by_region_year()
            self.logger.info(f"시군구 연령별 취업자 및 고용률 데이터 로드 완료: {len(employment_data)}개 레코드")
            
            # 5. 모든 데이터 통합
            final_data = self._merge_all_data(area_data, e9_data, industry_data, employment_data)
            
            # 6. 최종 데이터 정리
            final_data = self._clean_final_data(final_data)
            
            # 7. 결과 저장
            self._save_final_data(final_data)
            
            self.final_data = final_data
            self.logger.info("전체 데이터 처리 완료")
            
        except Exception as e:
            self.logger.error(f"데이터 처리 중 오류 발생: {e}")
            raise
    
    def _merge_all_data(self, area_data, e9_data, industry_data, employment_data):
        """모든 데이터 통합"""
        self.logger.info("데이터 통합 시작")
        
        # 1. 지역 데이터를 기준으로 모든 데이터 병합
        # E9 데이터와 병합 (left join으로 기본 지역 정보 유지)
        merged = pd.merge(area_data, e9_data, on=['시도', '시군구'], how='left')
        
        # 산업별 데이터와 병합 (left join으로 누락된 데이터 허용)
        merged = pd.merge(merged, industry_data, on=['시도', '시군구', '년도', '반기'], how='left')
        
        # 경제활동인구 데이터와 병합 (left join으로 누락된 데이터 허용)
        merged = pd.merge(merged, employment_data, on=['시도', '시군구', '년도', '반기'], how='left')
        
        self.logger.info(f"데이터 통합 완료: {len(merged)}개 레코드")
        return merged
    
    def _clean_final_data(self, df):
        """최종 데이터 정리"""
        self.logger.info("최종 데이터 정리 시작")
        
        # 필요한 칼럼만 선택하고 순서 정리
        final_columns = [
            '시도', '시군구', '면적', '년도', '반기',
            'E9_체류자수', '제조업_종사자수', '서비스업_종사자수', '취업자', '고용률'
        ]
        
        # 존재하지 않는 칼럼은 0으로 채우기
        for col in final_columns:
            if col not in df.columns:
                df[col] = 0
        
        # 최종 칼럼 순서로 정리
        df = df[final_columns].copy()
        
        # 결측값 처리
        df = df.fillna(0)
        
        # 연도가 0인 레코드 제거 및 2019년 이후 데이터만 유지
        df = df[(df['년도'] > 0) & (df['년도'] >= 2019)].copy()
        
        # 데이터 타입 정리
        df['년도'] = df['년도'].astype(int)
        df['반기'] = df['반기'].astype(int)
        df['면적'] = df['면적'].astype(float)
        df['E9_체류자수'] = df['E9_체류자수'].astype(float)
        df['제조업_종사자수'] = df['제조업_종사자수'].astype(float)
        df['서비스업_종사자수'] = df['서비스업_종사자수'].astype(float)
        df['취업자'] = df['취업자'].astype(float)
        df['고용률'] = df['고용률'].astype(float)
        
        # 시도, 시군구 순으로 정렬
        df = df.sort_values(['시도', '시군구', '년도', '반기']).reset_index(drop=True)
        
        self.logger.info(f"최종 데이터 정리 완료: {len(df)}개 레코드")
        return df
    
    def _save_final_data(self, df):
        """최종 데이터 저장"""
        self.logger.info("최종 데이터 저장 시작")
        
        try:
            # UTF-8 인코딩으로 저장
            df.to_csv(self.config.output_utf8, index=False, encoding='utf-8')
            self.logger.info(f"UTF-8 파일 저장 완료: {self.config.output_utf8}")
            
            # CP949 인코딩으로 저장
            df.to_csv(self.config.output_cp949, index=False, encoding='cp949')
            self.logger.info(f"CP949 파일 저장 완료: {self.config.output_cp949}")
            
        except Exception as e:
            self.logger.error(f"파일 저장 실패: {e}")
            raise
    
    def get_final_data(self):
        """최종 데이터 반환"""
        if self.final_data is None:
            self.process_all_data()
        return self.final_data
    
    def get_summary_stats(self):
        """데이터 요약 통계 반환"""
        df = self.get_final_data()
        
        summary = {
            '총_레코드_수': len(df),
            '지역_수': df[['시도', '시군구']].drop_duplicates().shape[0],
            '연도_범위': f"{df['년도'].min()} ~ {df['년도'].max()}",
            '반기': sorted(df['반기'].unique()),
            '시도_목록': sorted(df['시도'].unique())
        }
        
        return summary
