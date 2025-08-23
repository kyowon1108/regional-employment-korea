#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
취업자 및 고용률 데이터 전처리 클래스
시군구별 취업자 및 고용률 데이터를 처리하여 표준 형식으로 변환
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from .area_master import AreaMaster

class EmploymentProcessor:
    def __init__(self, data_path: str = "data/raw/시군구_연령별_취업자_및_고용률.csv"):
        self.data_path = Path(data_path)
        # 상대 경로 문제를 해결하기 위해 절대 경로 사용
        import os
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.area_master = AreaMaster("data/raw/지역_면적_utf8.csv")
        self.logger = self._setup_logger()
        
    def _setup_logger(self):
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def load_raw_data(self) -> pd.DataFrame:
        """원본 데이터를 로드합니다."""
        encodings = ['cp949', 'euc-kr', 'utf-8']
        
        for encoding in encodings:
            try:
                self.logger.info(f"인코딩 {encoding}으로 데이터 로드 시도")
                df = pd.read_csv(self.data_path, encoding=encoding)
                self.logger.info(f"인코딩 {encoding}으로 데이터 로드 성공: {df.shape}")
                return df
            except UnicodeDecodeError:
                self.logger.warning(f"인코딩 {encoding} 실패")
                continue
            except Exception as e:
                self.logger.error(f"데이터 로드 중 오류 발생: {e}")
                continue
        
        raise ValueError("모든 인코딩 시도 실패")
    
    def clean_column_names(self) -> pd.DataFrame:
        """칼럼명을 정리합니다."""
        df = self.load_raw_data()
        
        # 칼럼명 정리
        df = df.rename(columns={
            '행정구역별': '시도시군구',
            '연령별': '구분',
            '항목': '지표'
        })
        
        self.logger.info(f"칼럼명 정리 완료: {df.shape}")
        return df
    
    def filter_target_age_group(self) -> pd.DataFrame:
        """15-64세 연령대 데이터만 필터링합니다. 15-64세가 없으면 '계' 데이터를 사용합니다."""
        df = self.clean_column_names()
        
        # 먼저 15-64세 데이터를 시도
        target_age = '15 - 64세'
        df_15_64 = df[df['구분'] == target_age]
        
        # 15-64세가 없는 지역은 '계' 데이터를 사용
        regions_without_15_64 = []
        for region in df['시도시군구'].unique():
            region_data = df[df['시도시군구'] == region]
            if '15 - 64세' not in region_data['구분'].values and '계' in region_data['구분'].values:
                regions_without_15_64.append(region)
        
        if regions_without_15_64:
            self.logger.info(f"다음 지역은 '계' 데이터를 사용: {regions_without_15_64}")
            # '계' 데이터 추가
            df_gye = df[(df['시도시군구'].isin(regions_without_15_64)) & (df['구분'] == '계')]
            # '계'를 '15 - 64세'로 변경하여 통일
            df_gye = df_gye.copy()
            df_gye['구분'] = '15 - 64세'
            # 두 데이터프레임 합치기
            df = pd.concat([df_15_64, df_gye], ignore_index=True)
        else:
            df = df_15_64
        
        self.logger.info(f"15-64세 연령대 필터링 완료: {df.shape}")
        return df
    
    def extract_target_years(self, start_year: int = 2019, end_year: int = 2023) -> pd.DataFrame:
        """목표 연도의 데이터만 추출합니다."""
        df = self.filter_target_age_group()
        
        # 기본 칼럼들
        base_columns = ['시도시군구', '구분', '지표']
        
        # 연도별 칼럼 찾기 (2021~2023년)
        year_columns = []
        for year in range(2021, 2024):  # 실제 데이터는 2021~2023년만 있음
            for half in [1, 2]:
                col_name = f"{year}.{half}.2"
                if col_name in df.columns:
                    year_columns.append(col_name)
        
        # 필요한 칼럼만 선택
        selected_columns = base_columns + year_columns
        df = df[selected_columns]
        
        self.logger.info(f"목표 연도 추출 완료: {df.shape}, 선택된 연도 칼럼: {len(year_columns)}")
        return df
    
    def separate_sido_sigungu(self) -> pd.DataFrame:
        """시도시군구를 시도와 시군구로 분리합니다."""
        df = self.extract_target_years()
        
        # AreaMaster 초기화
        self.area_master.load_area_data()
        self.area_master.create_sido_sigungu_mapping()
        
        # 원본 데이터의 행 번호를 보존하기 위해 인덱스 리셋
        df = df.reset_index(drop=True)
        
        # 원본 데이터에서 고성군의 위치를 확인
        original_df = self.load_raw_data()
        goseong_indices = []
        for i, row in original_df.iterrows():
            if '고성' in str(row.iloc[0]):
                goseong_indices.append(i)
        
        self.logger.info(f"원본 데이터에서 고성군 위치: {goseong_indices}")
        
        # 시도와 시군구 분리
        def extract_sido_sigungu(region):
            if pd.isna(region):
                return None, None
            
            region_str = str(region)
            # "서울 종로구" -> "서울", "종로구"
            if " " in region_str:
                parts = region_str.split(" ", 1)
                if len(parts) == 2:
                    return parts[0], parts[1]
            
            # 시군구만 있는 경우 (예: "종로구")
            sigungu = region_str
            sido = self.area_master.get_sido_by_sigungu(sigungu)
            return sido, sigungu
        
        # 시도와 시군구 추출
        df[['시도', '시군구']] = df['시도시군구'].apply(
            lambda x: pd.Series(extract_sido_sigungu(x))
        )
        
        # 연기군 -> 세종특별자치시 매핑 (정규화 전에 처리)
        mask = df['시군구'] == '연기군'
        df.loc[mask, '시도'] = '세종특별자치시'
        df.loc[mask, '시군구'] = '세종특별자치시'
        
        # 군위군 특별 처리: 대구 군위군을 경상북도 군위군으로 수정 (정규화 전에 처리)
        mask = (df['시도'] == '대구') & (df['시군구'] == '군위군')
        df.loc[mask, '시도'] = '경상북도'
        
        # 시도 표준명 정규화
        normalize = {
            '서울':'서울특별시','부산':'부산광역시','대구':'대구광역시','인천':'인천광역시','광주':'광주광역시','대전':'대전광역시','울산':'울산광역시','세종':'세종특별자치시',
            '경기':'경기도','강원':'강원특별자치도','충북':'충청북도','충남':'충청남도','전북':'전라북도','전남':'전라남도','경북':'경상북도','경남':'경상남도','제주':'제주특별자치도'
        }
        df['시도'] = df['시도'].apply(lambda s: normalize.get(s, s))
        
        # 고성군 특별 처리: 원본 데이터의 행 번호를 기준으로 정확한 시도 매핑
        # goseong_indices에서 확인된 실제 행 번호를 기준으로 구분
        # 1766~1782행: 강원도 고성군
        # 3318~3334행: 경상남도 고성군
        goseong_mask = df['시군구'] == '고성군'
        if goseong_mask.any():
            goseong_df_indices = df[goseong_mask].index.tolist()
            
            # 고성군 데이터가 있는 경우, 순서에 따라 시도 결정
            # 첫 번째 블록 (처음 2개 행): 강원특별자치도
            # 두 번째 블록 (나중 2개 행): 경상남도
            if len(goseong_df_indices) >= 2:
                mid_point = len(goseong_df_indices) // 2
                
                # 첫 번째 절반: 강원특별자치도
                for i in range(mid_point):
                    idx = goseong_df_indices[i]
                    df.loc[idx, '시도'] = '강원특별자치도'
                
                # 두 번째 절반: 경상남도
                for i in range(mid_point, len(goseong_df_indices)):
                    idx = goseong_df_indices[i]
                    df.loc[idx, '시도'] = '경상남도'
            else:
                # 데이터가 1개만 있으면 강원특별자치도로 처리
                df.loc[goseong_mask, '시도'] = '강원특별자치도'
        
        # 원본 칼럼들 제거
        df = df.drop(['시도시군구', '구분'], axis=1)
        
        self.logger.info(f"시도-시군구 분리 완료: {df.shape}")
        return df
    
    def convert_to_long_format(self) -> pd.DataFrame:
        """와이드 포맷을 롱 포맷으로 변환합니다."""
        df = self.separate_sido_sigungu()
        
        # 연도 칼럼들 찾기 (시도, 시군구, 지표 제외)
        year_columns = [col for col in df.columns if col not in ['시도', '시군구', '지표']]
        
        # melt를 사용하여 롱 포맷으로 변환
        df_melted = df.melt(
            id_vars=['시도', '시군구', '지표'],
            value_vars=year_columns,
            var_name='연도_반기',
            value_name='값'
        )
        
        # 연도와 반기 분리 (예: "2021.1.2" -> 2021, 1)
        df_melted[['연도', '반기']] = df_melted['연도_반기'].str.extract(r'(\d{4})\.(\d+)\.2')
        df_melted['연도'] = df_melted['연도'].astype(int)
        df_melted['반기'] = df_melted['반기'].astype(int)
        
        # 연도_반기 칼럼 제거
        df_melted = df_melted.drop('연도_반기', axis=1)
        
        # 값을 숫자로 변환 (빈 값은 0으로)
        df_melted['값'] = pd.to_numeric(df_melted['값'], errors='coerce').fillna(0)
        
        self.logger.info(f"롱 포맷 변환 완료: {df_melted.shape}")
        return df_melted
    
    def pivot_employment_data(self) -> pd.DataFrame:
        """취업자와 고용률 데이터를 피벗하여 별도 칼럼으로 만듭니다."""
        df = self.convert_to_long_format()
        
        # 취업자와 고용률 데이터를 분리
        employment_data = df[df['지표'] == '취업자 (천명)'].copy()
        rate_data = df[df['지표'] == '고용률 (%)'].copy()
        
        # 취업자 데이터에서 지표 칼럼 제거하고 칼럼명 변경
        employment_data = employment_data.drop('지표', axis=1)
        employment_data = employment_data.rename(columns={'값': '취업자'})
        
        # 고용률 데이터에서 지표 칼럼 제거하고 칼럼명 변경
        rate_data = rate_data.drop('지표', axis=1)
        rate_data = rate_data.rename(columns={'값': '고용률'})
        
        # 시도, 시군구, 연도, 반기를 기준으로 병합
        result_df = employment_data.merge(
            rate_data, 
            on=['시도', '시군구', '연도', '반기'], 
            how='outer'
        )
        
        # NaN 값을 0으로 채우기
        result_df = result_df.fillna(0)
        
        # 취업자를 천명 단위에서 명 단위로 변환하고 정수로 변환
        result_df['취업자'] = (result_df['취업자'] * 1000).round().astype(int)
        
        self.logger.info(f"취업자/고용률 피벗 완료: {result_df.shape}")
        return result_df
    
    def aggregate_subdistricts_to_city(self, df: pd.DataFrame) -> pd.DataFrame:
        """시군구가 매핑과 불일치하는 하위 구 단위를 같은 시 단위로 합산합니다.
        - 합산: 취업자 합계, 고용률은 취업자 가중평균
        - 기준 매핑: data/processed/시도_시군구_매핑.csv
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
            return sigungu

        work = df.copy()
        work['표준시군구'] = work.apply(map_to_standard, axis=1)
        # 취업자 합, 고용률 가중평균
        grouped = (
            work.groupby(['시도','표준시군구','연도','반기'], as_index=False)
                .apply(lambda x: pd.Series({
                    '취업자': x['취업자'].sum(),
                    '고용률': 0 if x['취업자'].sum()==0 else (x['고용률']*x['취업자']).sum()/x['취업자'].sum()
                }))
                .reset_index(drop=True)
        )
        grouped = grouped.rename(columns={'표준시군구':'시군구'})
        grouped['취업자'] = grouped['취업자'].fillna(0).round().astype(int)
        grouped['고용률'] = grouped['고용률'].fillna(0)
        return grouped

    def process_data(self, start_year: int = 2019, end_year: int = 2023) -> pd.DataFrame:
        """전체 데이터 전처리 파이프라인을 실행합니다."""
        try:
            self.logger.info("취업자 및 고용률 데이터 전처리 시작")
            
            # 전체 파이프라인 실행
            result_df = self.pivot_employment_data()
            # 시 하위 구 합산 집계
            result_df = self.aggregate_subdistricts_to_city(result_df)
            
            self.logger.info(f"취업자 및 고용률 데이터 전처리 완료: {result_df.shape}")
            return result_df
            
        except Exception as e:
            self.logger.error(f"데이터 전처리 중 오류 발생: {e}")
            raise
    
    def export_data(self, df: pd.DataFrame, output_path: str = None, encoding: str = 'utf-8') -> str:
        """전처리된 데이터를 파일로 저장합니다."""
        if output_path is None:
            # 절대 경로 사용
            import os
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            output_path = os.path.join(base_dir, "data", "processed", f"취업자_고용률_전처리완료_{encoding}.csv")
        
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
    employment_processor = EmploymentProcessor()
    
    try:
        # 데이터 전처리 실행
        processed_data = employment_processor.process_data()
        
        # 요약 정보 출력
        summary = employment_processor.get_summary()
        print("=== 취업자 및 고용률 데이터 전처리 결과 ===")
        for key, value in summary.items():
            print(f"{key}: {value}")
        
        # 데이터 미리보기
        print(f"\n=== 데이터 미리보기 ===")
        print(processed_data.head())
        
        # 결과 내보내기
        employment_processor.export_processed_data()
        
    except Exception as e:
        print(f"전처리 실패: {e}")
