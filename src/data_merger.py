#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
데이터 병합 클래스
전처리된 모든 데이터를 시도-시군구-연도-반기 기준으로 병합
"""

import pandas as pd
import logging
from typing import Dict, List, Optional
from pathlib import Path
from .area_master import AreaMaster
from .e9_processor import E9Processor
from .industry_processor import IndustryProcessor
from .employment_processor import EmploymentProcessor


class DataMerger:
    """전처리된 모든 데이터를 병합하는 클래스"""
    
    def __init__(self):
        """데이터 병합기를 초기화합니다."""
        # 상대 경로 문제를 해결하기 위해 절대 경로 사용
        import os
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        self.area_master = AreaMaster(os.path.join(base_dir, "data", "raw", "지역_면적_utf8.csv"))
        self.e9_processor = E9Processor(os.path.join(base_dir, "data", "raw", "시군구별_E9_체류자(2014~2023).csv"))
        # 새로운 산업별 고용 데이터 사용
        self.industry_processor = IndustryProcessor(os.path.join(base_dir, "data", "raw", "산업별 고용 시군구 2025-08-22.csv"))
        self.employment_processor = EmploymentProcessor(os.path.join(base_dir, "data", "raw", "시군구_연령별_취업자_및_고용률.csv"))
        self.merged_data = None
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
    
    def process_all_data(self, start_year: int = 2019, end_year: int = 2023) -> Dict[str, pd.DataFrame]:
        """모든 데이터 전처리 실행"""
        try:
            self.logger.info("모든 데이터 전처리 시작")
            
            # 1. 지역 마스터 데이터 처리
            self.logger.info("1단계: 지역 마스터 데이터 처리")
            area_data = self.area_master.load_area_data()
            area_mapping = self.area_master.create_sido_sigungu_mapping()
            
            # 2. E9 체류자 데이터 전처리
            self.logger.info("2단계: E9 체류자 데이터 전처리")
            e9_data = self.e9_processor.process_data(start_year, end_year)
            
            # 3. 산업별 종사자 데이터 전처리
            self.logger.info("3단계: 산업별 종사자 데이터 전처리")
            industry_data = self.industry_processor.process_data(start_year, end_year)
            
            # 4. 취업자 및 고용률 데이터 전처리
            self.logger.info("4단계: 취업자 및 고용률 데이터 전처리")
            # employment_data = self.employment_processor.process_data(start_year, end_year)
            # 이미 전처리된 파일을 직접 읽기
            import os
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            employment_file = os.path.join(base_dir, "data", "processed", "취업자_고용률_전처리완료_utf-8.csv")
            employment_data = pd.read_csv(employment_file)
            
            processed_data = {
                'area': area_data,
                'e9': e9_data,
                'industry': industry_data,
                'employment': employment_data
            }
            
            self.logger.info("모든 데이터 전처리 완료")
            return processed_data
            
        except Exception as e:
            self.logger.error(f"데이터 전처리 과정에서 오류 발생: {e}")
            raise
    
    def create_base_structure(self, start_year: int = 2019, end_year: int = 2023) -> pd.DataFrame:
        """기본 데이터 구조 생성 (모든 지역-연도-반기 조합)"""
        self.logger.info("기본 데이터 구조 생성 시작")
        
        # 모든 지역 정보 가져오기
        regions = self.area_master.get_all_regions()
        
        # 모든 연도-반기 조합 생성
        year_half_combinations = []
        for year in range(start_year, end_year + 1):
            for half in [1, 2]:
                year_half_combinations.append((year, half))
        
        # 기본 구조 생성
        base_data = []
        for sido, sigungu, area in regions:
            for year, half in year_half_combinations:
                base_data.append({
                    '시도': sido,
                    '시군구': sigungu,
                    '면적': area,
                    '연도': year,
                    '반기': half
                })
        
        base_df = pd.DataFrame(base_data)
        self.logger.info(f"기본 데이터 구조 생성 완료: {len(base_df)}행")
        return base_df
    
    def merge_all_data(self) -> pd.DataFrame:
        """모든 전처리된 데이터를 병합합니다."""
        # 기본 데이터 구조 생성
        base_data = self.create_base_structure()
        
        # 모든 데이터 전처리
        processed_data = self.process_all_data()
        
        # E9 체류자 데이터 병합
        self.logger.info("E9 체류자 데이터 병합")
        merged_data = base_data.merge(
            processed_data['e9'],
            on=['시도', '시군구', '연도', '반기'],
            how='left'
        )
        
        # 산업별 종사자 데이터 병합
        self.logger.info("산업별 종사자 데이터 병합")
        merged_data = merged_data.merge(
            processed_data['industry'],
            on=['시도', '시군구', '연도', '반기'],
            how='left'
        )
        
        # 취업자 및 고용률 데이터 병합
        self.logger.info("취업자 및 고용률 데이터 병합")
        merged_data = merged_data.merge(
            processed_data['employment'],
            on=['시도', '시군구', '연도', '반기'],
            how='left'
        )
        
        self.logger.info(f"데이터 병합 완료: {len(merged_data)}행")
        
        # 내부 상태에 반영
        self.merged_data = merged_data
        
        # 결측값 처리 (내부 상태 기반)
        self.handle_missing_data()
        
        # 숫자 데이터를 정수로 변환
        numeric_columns = ['E9_체류자수', '제조업_종사자수', '서비스업_종사자수', '취업자']
        for col in numeric_columns:
            if col in self.merged_data.columns:
                self.merged_data[col] = self.merged_data[col].fillna(0).round().astype(int)
        
        return self.merged_data
    
    def handle_missing_data(self) -> pd.DataFrame:
        """결측값 처리"""
        if self.merged_data is None:
            self.logger.error("병합된 데이터가 없습니다")
            return None
        
        self.logger.info("결측값 처리 시작")
        
        # 숫자형 칼럼의 결측값을 0으로 채우기
        numeric_columns = [
            'E9_체류자수', '제조업_종사자수', '서비스업_종사자수',
            '취업자', '고용률'
        ]
        
        for col in numeric_columns:
            if col in self.merged_data.columns:
                missing_count = self.merged_data[col].isna().sum()
                if missing_count > 0:
                    self.logger.info(f"{col} 결측값 {missing_count}개를 0으로 채움")
                    self.merged_data[col] = self.merged_data[col].fillna(0)
        
        # 면적 칼럼의 결측값을 0으로 채우기
        if '면적' in self.merged_data.columns:
            missing_area = self.merged_data['면적'].isna().sum()
            if missing_area > 0:
                self.logger.info(f"면적 결측값 {missing_area}개를 0으로 채움")
                self.merged_data['면적'] = self.merged_data['면적'].fillna(0)
        
        self.logger.info("결측값 처리 완료")
        return self.merged_data
    
    def validate_final_data(self) -> bool:
        """최종 데이터 유효성 검사"""
        if self.merged_data is None:
            return False
        
        self.logger.info("최종 데이터 유효성 검사 시작")
        
        # 필수 칼럼 확인
        required_columns = ['시도', '시군구', '면적', '연도', '반기']
        missing_columns = [col for col in required_columns if col not in self.merged_data.columns]
        
        if missing_columns:
            self.logger.error(f"필수 칼럼 누락: {missing_columns}")
            return False
        
        # 데이터 타입 확인
        if not pd.api.types.is_integer_dtype(self.merged_data['연도']):
            self.logger.warning("연도 칼럼이 정수형이 아님")
        
        if not pd.api.types.is_integer_dtype(self.merged_data['반기']):
            self.logger.warning("반기 칼럼이 정수형이 아님")
        
        if not pd.api.types.is_numeric_dtype(self.merged_data['면적']):
            self.logger.warning("면적 칼럼이 숫자형이 아님")
        
        # 중복 데이터 확인
        duplicates = self.merged_data.duplicated(subset=['시도', '시군구', '연도', '반기'])
        if duplicates.any():
            duplicate_count = duplicates.sum()
            self.logger.warning(f"중복 데이터 발견: {duplicate_count}개")
        
        # 데이터 완성도 확인
        total_records = len(self.merged_data)
        complete_records = self.merged_data.dropna().shape[0]
        completion_rate = (complete_records / total_records) * 100
        
        self.logger.info(f"데이터 완성도: {completion_rate:.2f}% ({complete_records}/{total_records})")
        
        self.logger.info("최종 데이터 유효성 검사 완료")
        return True
    
    def get_final_summary(self) -> Dict:
        """최종 병합된 데이터 요약 정보"""
        if self.merged_data is None:
            return {}
        
        summary = {
            '총_행_수': len(self.merged_data),
            '시도_수': self.merged_data['시도'].nunique(),
            '시군구_수': self.merged_data['시군구'].nunique(),
            '연도_범위': f"{self.merged_data['연도'].min()}~{self.merged_data['연도'].max()}",
            '반기_종류': sorted(self.merged_data['반기'].unique()),
            '면적_총합': self.merged_data['면적'].sum(),
            '면적_평균': self.merged_data['면적'].mean()
        }
        
        # 각 데이터 칼럼별 요약 추가
        data_columns = [
            'E9_체류자수', '제조업_종사자수', '서비스업_종사자수',
            '취업자', '고용률'
        ]
        
        for col in data_columns:
            if col in self.merged_data.columns:
                summary[f'{col}_총합'] = self.merged_data[col].sum()
                summary[f'{col}_평균'] = self.merged_data[col].mean()
                summary[f'{col}_결측값_수'] = self.merged_data[col].isna().sum()
        
        return summary
    
    def export_final_data(self, output_path: str = "data/processed/최종_통합데이터.csv") -> str:
        """최종 병합된 데이터를 CSV로 내보내기"""
        if self.merged_data is None:
            self.logger.error("병합된 데이터가 없습니다")
            return None
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.merged_data.to_csv(output_path, index=False, encoding='utf-8')
        self.logger.info(f"최종 데이터 내보내기 완료: {output_path}")
        
        return str(output_path)
    
    def export_final_data_both(self, base_filename: str = "최종_통합데이터") -> dict:
        """최종 병합 데이터를 UTF-8과 CP949 두 포맷으로 모두 내보냅니다."""
        if self.merged_data is None:
            self.logger.error("병합된 데이터가 없습니다")
            return {}
        from pathlib import Path
        import os
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        out_dir = Path(os.path.join(base_dir, 'data', 'processed'))
        out_dir.mkdir(parents=True, exist_ok=True)
        results = {}
        utf8_path = out_dir / f"{base_filename}_utf-8.csv"
        cp949_path = out_dir / f"{base_filename}_cp949.csv"
        self.merged_data.to_csv(utf8_path, index=False, encoding='utf-8')
        self.merged_data.to_csv(cp949_path, index=False, encoding='cp949')
        self.logger.info(f"최종 데이터 내보내기 완료(UTF-8): {utf8_path}")
        self.logger.info(f"최종 데이터 내보내기 완료(CP949): {cp949_path}")
        results['utf8'] = str(utf8_path)
        results['cp949'] = str(cp949_path)
        return results
    
    def run_complete_pipeline(self, start_year: int = 2019, end_year: int = 2023) -> pd.DataFrame:
        """전체 파이프라인 실행"""
        try:
            self.logger.info("전체 데이터 전처리 파이프라인 시작")
            
            # 1. 데이터 병합 (내부 상태에 반영됨)
            merged_data = self.merge_all_data()
            
            # 2. 결측값 처리 (내부 상태 기반)
            processed_data = self.handle_missing_data()
            
            # 3. 최종 검증
            is_valid = self.validate_final_data()
            
            if not is_valid:
                self.logger.warning("데이터 유효성 검사에서 문제 발견")
            
            # 4. 최종 데이터 내보내기 (UTF-8, CP949)
            out = self.export_final_data_both()
            
            # 5. 요약 정보 출력
            summary = self.get_final_summary()
            self.logger.info("=== 최종 데이터 요약 ===")
            for key, value in summary.items():
                self.logger.info(f"{key}: {value}")
            
            self.logger.info("전체 파이프라인 완료")
            return processed_data
            
        except Exception as e:
            self.logger.error(f"파이프라인 실행 중 오류 발생: {e}")
            raise


if __name__ == "__main__":
    # 테스트 코드
    data_merger = DataMerger()
    
    try:
        # 전체 파이프라인 실행
        final_data = data_merger.run_complete_pipeline()
        
        # 데이터 미리보기
        print(f"\n=== 최종 데이터 미리보기 ===")
        print(final_data.head())
        
        print(f"\n=== 데이터 형태 ===")
        print(f"행 수: {len(final_data)}")
        print(f"열 수: {len(final_data.columns)}")
        print(f"칼럼: {list(final_data.columns)}")
        
    except Exception as e:
        print(f"파이프라인 실행 실패: {e}")
