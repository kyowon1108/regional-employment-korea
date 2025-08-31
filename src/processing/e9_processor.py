#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
E9 체류자 데이터 전처리 클래스
시군구별 E9 체류자 데이터를 처리하여 표준 형식으로 변환
"""

import pandas as pd
import logging
from typing import Dict, List, Optional
from pathlib import Path
from .area_master import AreaMaster


class E9Processor:
    """E9 체류자 데이터를 전처리하는 클래스"""
    
    def __init__(self, data_path: str = "data/raw/시군구별_E9_체류자(2014~2023).csv"):
        """
        Args:
            data_path (str): E9 체류자 데이터 파일 경로
        """
        self.data_path = Path(data_path)
        self.raw_data = None
        self.processed_data = None
        self.area_master = AreaMaster()
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
    
    def load_raw_data(self) -> pd.DataFrame:
        """원본 E9 체류자 데이터 로드 (인코딩 문제 해결)"""
        try:
            self.logger.info(f"E9 체류자 데이터 로드 중: {self.data_path}")
            
            # 여러 인코딩 시도
            encodings = ['cp949', 'euc-kr', 'utf-8']
            for encoding in encodings:
                try:
                    self.raw_data = pd.read_csv(self.data_path, encoding=encoding)
                    self.logger.info(f"인코딩 {encoding}으로 데이터 로드 성공")
                    break
                except UnicodeDecodeError:
                    continue
                except Exception as e:
                    self.logger.warning(f"인코딩 {encoding} 시도 실패: {e}")
                    continue
            
            if self.raw_data is None:
                raise ValueError("모든 인코딩 시도 실패")
            
            self.logger.info(f"데이터 로드 완료: {len(self.raw_data)}행")
            return self.raw_data
            
        except Exception as e:
            self.logger.error(f"데이터 로드 실패: {e}")
            raise
    
    def clean_column_names(self) -> pd.DataFrame:
        """칼럼명 정리"""
        if self.raw_data is None:
            self.load_raw_data()
        
        # 첫 번째 행을 헤더로 사용
        self.raw_data.columns = self.raw_data.iloc[0]
        self.raw_data = self.raw_data.iloc[1:].reset_index(drop=True)
        
        # 연도 칼럼명 설정 (2014~2023)
        year_columns = [str(year) for year in range(2014, 2024)]
        for i, year in enumerate(year_columns):
            if i + 3 < len(self.raw_data.columns):  # 3은 시도, 시군구, 구분 칼럼
                self.raw_data.columns.values[i + 3] = year
        
        # 칼럼명 정리
        column_mapping = {
            '행정구역(시군구)별(1)': '시도',
            '행정구역(시군구)별(2)': '시군구',
            '성별(1)': '구분'
        }
        
        # 존재하는 칼럼만 매핑
        for old_name, new_name in column_mapping.items():
            if old_name in self.raw_data.columns:
                self.raw_data = self.raw_data.rename(columns={old_name: new_name})
        
        self.logger.info("칼럼명 정리 완료")
        return self.raw_data
    
    def extract_target_years(self, start_year: int = 2019, end_year: int = 2023) -> pd.DataFrame:
        """목표 연도 데이터만 추출"""
        if self.raw_data is None:
            self.clean_column_names()
        
        # 먼저 '계' 데이터만 필터링
        total_data = self.raw_data[self.raw_data['구분'] == '계'].copy()
        
        # 연도별 반기별 칼럼 찾기
        year_columns = []
        for year in range(start_year, end_year + 1):
            column_name = str(year)
            if column_name in total_data.columns:
                year_columns.append(column_name)
        
        if not year_columns:
            self.logger.warning(f"목표 연도 칼럼을 찾을 수 없음: {start_year}~{end_year}")
            return total_data
        
        # 필요한 칼럼만 선택
        selected_columns = ['시도', '시군구', '구분'] + year_columns
        total_data = total_data[selected_columns]
        
        self.logger.info(f"목표 연도 데이터 추출 완료: {year_columns}")
        return total_data
    
    def filter_total_data(self) -> pd.DataFrame:
        """총계 데이터만 필터링합니다."""
        df = self.extract_target_years()
        
        # '구분' 칼럼 제거 (이미 '계' 데이터만 필터링됨)
        if '구분' in df.columns:
            df = df.drop('구분', axis=1)
        
        self.logger.info(f"총계 데이터 필터링 완료: {len(df)}행")
        return df
    
    def add_missing_sido(self) -> pd.DataFrame:
        """누락된 시도 정보를 추가합니다."""
        if self.raw_data is None:
            self.filter_total_data()
        
        # '총계', '소계' 데이터 제거
        self.raw_data = self.raw_data[
            (self.raw_data['시도'] != '총계') & 
            (self.raw_data['시군구'] != '소계')
        ]
        
        # 시도 정보가 없는 경우 처리
        missing_sido_mask = self.raw_data['시도'].isna() | (self.raw_data['시도'] == '')
        
        for idx in self.raw_data[missing_sido_mask].index:
            sigungu = self.raw_data.loc[idx, '시군구']
            if pd.notna(sigungu):
                sido = self.area_master.get_sido_by_sigungu(sigungu)
                if sido != "알수없음":
                    self.raw_data.loc[idx, '시도'] = sido
                    self.logger.debug(f"시도 정보 추가: {sigungu} -> {sido}")
        
        # 중복 데이터 제거 (시도, 시군구 기준)
        self.raw_data = self.raw_data.drop_duplicates(subset=['시도', '시군구'], keep='first')
        
        self.logger.info("누락된 시도 정보 추가 완료")
        return self.raw_data
    
    def convert_to_long_format(self) -> pd.DataFrame:
        """wide format을 long format으로 변환"""
        if self.raw_data is None:
            self.add_missing_sido()
        
        # 목표 연도 칼럼만 선택 (시도, 시군구 제외)
        year_columns = [col for col in self.raw_data.columns 
                        if col not in ['시도', '시군구'] and col.isdigit()]
        
        # 시도 표준명 정규화
        normalize = {
            '서울':'서울특별시','부산':'부산광역시','대구':'대구광역시','인천':'인천광역시','광주':'광주광역시','대전':'대전광역시','울산':'울산광역시','세종':'세종특별자치시',
            '경기':'경기도','강원':'강원특별자치도','충북':'충청북도','충남':'충청남도','전북':'전라북도','전남':'전라남도','경북':'경상북도','경남':'경상남도','제주':'제주특별자치도','제주도':'제주특별자치도'
        }
        if '시도' in self.raw_data.columns:
            self.raw_data['시도'] = self.raw_data['시도'].apply(lambda s: normalize.get(s, s))
        
        # 군위군 특별 처리: 대구광역시 군위군을 경상북도 군위군으로 수정
        if '시군구' in self.raw_data.columns:
            mask = (self.raw_data['시도'] == '대구광역시') & (self.raw_data['시군구'] == '군위군')
            self.raw_data.loc[mask, '시도'] = '경상북도'
        
        # 대구광역시 군위군 행 완전 제거 (잘못된 데이터)
        if '시군구' in self.raw_data.columns:
            mask = (self.raw_data['시도'] == '대구광역시') & (self.raw_data['시군구'] == '군위군')
            self.raw_data = self.raw_data[~mask]
        
        # 2019~2023년만 필터링
        target_years = ['2019', '2020', '2021', '2022', '2023']
        year_columns = [col for col in year_columns if col in target_years]
        
        # melt를 사용하여 long format으로 변환
        long_data = self.raw_data.melt(
            id_vars=['시도', '시군구'],
            value_vars=year_columns,
            var_name='연도',
            value_name='E9_체류자수'
        )
        
        # 연도를 정수형으로 변환
        long_data['연도'] = long_data['연도'].astype(int)
        
        # 반기 칼럼 추가 (기본값 2)
        long_data['반기'] = 2
        
        # E9_체류자수를 숫자형으로 변환하고 정수로 변환
        long_data['E9_체류자수'] = pd.to_numeric(long_data['E9_체류자수'], errors='coerce')
        long_data['E9_체류자수'] = long_data['E9_체류자수'].fillna(0).round().astype(int)
        
        # 결측값 확인
        missing_count = long_data['E9_체류자수'].isna().sum()
        if missing_count > 0:
            self.logger.warning(f"E9_체류자수 결측값: {missing_count}개")
        
        self.logger.info(f"Long format 변환 완료: {len(long_data)}행")
        return long_data
    
    def aggregate_subdistricts_to_city(self, df: pd.DataFrame) -> pd.DataFrame:
        """시군구가 매핑과 불일치하는 하위 구 단위를 같은 시 단위로 합산합니다.
        - 기준: data/processed/시도_시군구_매핑.csv 의 (시도, 시군구)
        - 불일치 항목의 시군구에서 첫 공백 전까지를 기준 시군구로 사용 (예: '천안시 서북구' -> '천안시')
        - 기준 시군구가 매핑에 존재할 때만 치환/집계 수행
        """
        try:
            import os
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            mapping_path = os.path.join(base_dir, 'data', 'processed', '시도_시군구_매핑.csv')
            mapping_df = pd.read_csv(mapping_path)
            valid_pairs = set(zip(mapping_df['시도'], mapping_df['시군구']))
        except Exception as e:
            self.logger.warning(f"시도_시군구_매핑.csv 로드 실패, 집계 스킵: {e}")
            return df

        def to_base_city(s: str) -> str:
            if not isinstance(s, str):
                return s
            parts = s.split(' ')
            return parts[0] if len(parts) > 1 else s

        def map_to_standard(row):
            sido = row['시도']
            sigungu = row['시군구']
            if (sido, sigungu) in valid_pairs:
                return sigungu
            candidate = to_base_city(sigungu)
            if (sido, candidate) in valid_pairs:
                return candidate
            return sigungu  # 매핑 불가 시 원본 유지

        work = df.copy()
        work['표준시군구'] = work.apply(map_to_standard, axis=1)
        # 그룹 집계
        grouped = (
            work.groupby(['시도', '표준시군구', '연도', '반기'], as_index=False)['E9_체류자수']
                .sum()
        )
        grouped = grouped.rename(columns={'표준시군구': '시군구'})
        # 정수 보장
        grouped['E9_체류자수'] = grouped['E9_체류자수'].fillna(0).round().astype(int)
        return grouped

    def process_data(self, start_year: int = 2019, end_year: int = 2023) -> pd.DataFrame:
        """전체 전처리 과정 실행"""
        try:
            self.logger.info("E9 체류자 데이터 전처리 시작")
            
            # 단계별 처리
            self.load_raw_data()
            self.clean_column_names()
            self.extract_target_years(start_year, end_year)
            self.filter_total_data()
            self.add_missing_sido()
            long_df = self.convert_to_long_format()
            # 시 하위 구 합산 집계
            long_df = self.aggregate_subdistricts_to_city(long_df)
            self.processed_data = long_df
            
            # 최종 데이터 검증
            self._validate_processed_data()
            
            self.logger.info("E9 체류자 데이터 전처리 완료")
            return self.processed_data
            
        except Exception as e:
            self.logger.error(f"전처리 과정에서 오류 발생: {e}")
            raise
    
    def _validate_processed_data(self) -> bool:
        """전처리된 데이터 유효성 검사"""
        if self.processed_data is None:
            return False
        
        # 필수 칼럼 확인
        required_columns = ['시도', '시군구', '연도', '반기', 'E9_체류자수']
        missing_columns = [col for col in required_columns if col not in self.processed_data.columns]
        
        if missing_columns:
            self.logger.error(f"필수 칼럼 누락: {missing_columns}")
            return False
        
        # 데이터 타입 확인
        if not pd.api.types.is_integer_dtype(self.processed_data['연도']):
            self.logger.warning("연도 칼럼이 정수형이 아님")
        
        if not pd.api.types.is_numeric_dtype(self.processed_data['E9_체류자수']):
            self.logger.warning("E9_체류자수 칼럼이 숫자형이 아님")
        
        # 결측값 확인
        missing_count = self.processed_data['E9_체류자수'].isna().sum()
        if missing_count > 0:
            self.logger.warning(f"E9_체류자수 결측값: {missing_count}개")
        
        self.logger.info("전처리된 데이터 유효성 검사 완료")
        return True
    
    def get_summary(self) -> Dict:
        """전처리된 데이터 요약 정보"""
        if self.processed_data is None:
            return {}
        
        summary = {
            '총_행_수': len(self.processed_data),
            '시도_수': self.processed_data['시도'].nunique(),
            '시군구_수': self.processed_data['시군구'].nunique(),
            '연도_범위': f"{self.processed_data['연도'].min()}~{self.processed_data['연도'].max()}",
            'E9_체류자수_총합': self.processed_data['E9_체류자수'].sum(),
            'E9_체류자수_평균': self.processed_data['E9_체류자수'].mean(),
            '결측값_수': self.processed_data['E9_체류자수'].isna().sum()
        }
        
        return summary
    
    def export_processed_data(self, output_path: str = "data/processed/E9_체류자_전처리완료.csv"):
        """전처리된 데이터를 CSV로 내보내기"""
        if self.processed_data is None:
            self.logger.error("전처리된 데이터가 없습니다")
            return None
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.processed_data.to_csv(output_path, index=False, encoding='utf-8')
        self.logger.info(f"전처리된 데이터 내보내기 완료: {output_path}")
        
        return output_path
    
    def export_data(self, df: pd.DataFrame, output_path: str = None, encoding: str = 'utf-8') -> str:
        """전처리된 데이터를 파일로 저장합니다."""
        if output_path is None:
            # 절대 경로 사용
            import os
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            output_path = os.path.join(base_dir, "data", "processed", f"E9_체류자_전처리완료_{encoding}.csv")
        
        # 디렉토리 생성
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # 파일 저장
        df.to_csv(output_path, index=False, encoding=encoding)
        
        self.logger.info(f"데이터 저장 완료: {output_path}")
        return output_path
    
    def export_both_encodings(self, df: pd.DataFrame) -> dict:
        """UTF-8과 CP949 인코딩으로 모두 저장합니다."""
        results = {}
        
        # UTF-8로 저장
        utf8_path = self.export_data(df, encoding='utf-8')
        results['utf8'] = utf8_path
        
        # CP949로 저장
        cp949_path = self.export_data(df, encoding='cp949')
        results['cp949'] = cp949_path
        
        return results


if __name__ == "__main__":
    # 테스트 코드
    e9_processor = E9Processor()
    
    try:
        # 데이터 전처리 실행
        processed_data = e9_processor.process_data()
        
        # 요약 정보 출력
        summary = e9_processor.get_summary()
        print("=== E9 체류자 데이터 전처리 결과 ===")
        for key, value in summary.items():
            print(f"{key}: {value}")
        
        # 데이터 미리보기
        print(f"\n=== 데이터 미리보기 ===")
        print(processed_data.head())
        
        # 결과 내보내기
        e9_processor.export_processed_data()
        
    except Exception as e:
        print(f"전처리 실패: {e}")
