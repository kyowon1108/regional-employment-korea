#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
산업별 종사자 데이터 전처리 클래스
지역별 산업별 종사자 데이터를 처리하여 제조업과 서비스업 종사자수 추출
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from .area_master import AreaMaster

class IndustryProcessor:
    def __init__(self, data_path: str = "data/raw/산업별 고용 시군구 2025-08-22.csv"):
        self.data_path = Path(data_path)
        # 상대 경로 문제를 해결하기 위해 절대 경로 사용
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
                # 첫 번째 행을 헤더로 사용하고, 두 번째 행은 건너뛰기
                df = pd.read_csv(self.data_path, encoding=encoding, header=0, skiprows=[1])
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
        
        # 첫 번째 두 칼럼은 이미 올바름
        # 지역별 -> 지역, 산업별 -> 산업
        df = df.rename(columns={
            '지역별': '지역',
            '산업별': '산업'
        })
        
        self.logger.info(f"칼럼명 정리 완료: {df.shape}")
        return df
    
    def filter_manufacturing_service(self) -> pd.DataFrame:
        """제조업과 서비스업 데이터만 필터링합니다."""
        df = self.clean_column_names()
        
        # 제조업과 서비스업만 필터링
        target_industries = ['광업.제조업(B,C)', '사업ㆍ개인ㆍ공공서비스업(E,L~S)']
        df = df[df['산업'].isin(target_industries)]
        
        self.logger.info(f"제조업/서비스업 필터링 완료: {df.shape}")
        return df
    
    def extract_target_years(self, start_year: int = 2019, end_year: int = 2023) -> pd.DataFrame:
        """목표 연도의 데이터만 추출합니다."""
        df = self.filter_manufacturing_service()
        
        # 기본 칼럼들
        base_columns = ['지역', '산업']
        
        # 연도별 칼럼 찾기 (전체종사자 (명) 칼럼만)
        year_columns = []
        for year in range(start_year, end_year + 1):
            for half in [1, 2]:
                col_name = f"{year} {half}/2"
                if col_name in df.columns:
                    year_columns.append(col_name)
        
        # 필요한 칼럼만 선택
        selected_columns = base_columns + year_columns
        df = df[selected_columns]
        
        self.logger.info(f"목표 연도 추출 완료: {df.shape}, 선택된 연도 칼럼: {len(year_columns)}")
        return df
    
    def separate_sido_sigungu(self) -> pd.DataFrame:
        """지역을 시도와 시군구로 분리합니다."""
        df = self.extract_target_years()
        
        # AreaMaster 초기화
        self.area_master.load_area_data()
        self.area_master.create_sido_sigungu_mapping()
        
        # 시도와 시군구 분리
        def extract_sido_sigungu(region):
            if pd.isna(region):
                return None, None
            
            region_str = str(region)
            # "서울특별시 종로구" 또는 "제주도 제주시" -> "서울특별시", "종로구" 또는 "제주도", "제주시"
            if " " in region_str:
                parts = region_str.split(" ", 1)
                if len(parts) == 2:
                    sido = parts[0]
                    sigungu = parts[1]
                    # 제주도는 특별자치도로 변경하지 않음 (원본 유지)
                    return sido, sigungu
            
            # 시군구만 있는 경우 (예: "종로구")
            sigungu = region_str
            sido = self.area_master.get_sido_by_sigungu(sigungu)
            return sido, sigungu
        
        # 시도와 시군구 추출
        df[['시도', '시군구']] = df['지역'].apply(
            lambda x: pd.Series(extract_sido_sigungu(x))
        )
        
        # 시도 표준명 정규화
        normalize = {
            '서울':'서울특별시','부산':'부산광역시','대구':'대구광역시','인천':'인천광역시','광주':'광주광역시','대전':'대전광역시','울산':'울산광역시','세종':'세종특별자치시',
            '경기':'경기도','강원':'강원특별자치도','충북':'충청북도','충남':'충청남도','전북':'전라북도','전남':'전라남도','경북':'경상북도','경남':'경상남도','제주':'제주특별자치도','제주도':'제주특별자치도'
        }
        df['시도'] = df['시도'].apply(lambda s: normalize.get(s, s))
        
        # 군위군 특별 처리: 대구광역시 군위군을 경상북도 군위군으로 수정
        mask = (df['시도'] == '대구광역시') & (df['시군구'] == '군위군')
        df.loc[mask, '시도'] = '경상북도'
        
        # 대구광역시 군위군 행 완전 제거 (잘못된 데이터)
        mask = (df['시도'] == '대구광역시') & (df['시군구'] == '군위군')
        df = df[~mask]
        
        # 산업 정보는 보존 (지역 칼럼만 제거)
        df = df.drop(['지역'], axis=1)
        
        self.logger.info(f"시도-시군구 분리 완료: {df.shape}")
        return df
    
    def convert_to_long_format(self) -> pd.DataFrame:
        """와이드 포맷을 롱 포맷으로 변환합니다."""
        df = self.separate_sido_sigungu()
        
        # 연도 칼럼들 찾기 (시도, 시군구 제외)
        year_columns = [col for col in df.columns if col not in ['시도', '시군구']]
        
        # 각 산업별로 별도로 처리
        result_data = []
        
        for _, row in df.iterrows():
            sido = row['시도']
            sigungu = row['시군구']
            
            # 연도별 데이터 추출
            for year_col in year_columns:
                # 연도와 반기 추출 (예: "2019 1/2" -> 2019, 1)
                if ' ' in year_col and '/' in year_col:
                    year_part, half_part = year_col.split(' ', 1)
                    year = int(year_part)
                    half = int(half_part.split('/')[0])
                    
                    # 종사자수 추출
                    employees = row[year_col]
                    if pd.notna(employees) and employees != '':
                        try:
                            employees = float(employees)
                        except:
                            employees = 0.0
                    else:
                        employees = 0.0
                    
                    result_data.append({
                        '시도': sido,
                        '시군구': sigungu,
                        '연도': year,
                        '반기': half,
                        '산업': '광업.제조업(B,C)' if '광업.제조업(B,C)' in str(row.get('산업', '')) else '사업ㆍ개인ㆍ공공서비스업(E,L~S)',
                        '종사자수': employees
                    })
        
        result_df = pd.DataFrame(result_data)
        
        # 종사자수를 숫자로 변환
        result_df['종사자수'] = pd.to_numeric(result_df['종사자수'], errors='coerce').fillna(0)
        
        self.logger.info(f"롱 포맷 변환 완료: {result_df.shape}")
        return result_df
    
    def pivot_industry_data(self) -> pd.DataFrame:
        """산업별 데이터를 피벗하여 제조업과 서비스업 종사자수를 별도 칼럼으로 만듭니다."""
        df = self.convert_to_long_format()
        
        # 제조업과 서비스업 데이터를 분리
        manufacturing_data = df[df['산업'] == '광업.제조업(B,C)'].copy()
        service_data = df[df['산업'] == '사업ㆍ개인ㆍ공공서비스업(E,L~S)'].copy()
        
        # 제조업 데이터에서 산업 칼럼 제거하고 칼럼명 변경
        manufacturing_data = manufacturing_data.drop('산업', axis=1)
        manufacturing_data = manufacturing_data.rename(columns={'종사자수': '제조업_종사자수'})
        
        # 서비스업 데이터에서 산업 칼럼 제거하고 칼럼명 변경
        service_data = service_data.drop('산업', axis=1)
        service_data = service_data.rename(columns={'종사자수': '서비스업_종사자수'})
        
        # 시도, 시군구, 연도, 반기를 기준으로 병합
        result_df = manufacturing_data.merge(
            service_data, 
            on=['시도', '시군구', '연도', '반기'], 
            how='outer'
        )
        
        # NaN 값을 0으로 채우기
        result_df = result_df.fillna(0)
        
        # 종사자수를 정수로 변환
        result_df['제조업_종사자수'] = result_df['제조업_종사자수'].round().astype(int)
        result_df['서비스업_종사자수'] = result_df['서비스업_종사자수'].round().astype(int)
        
        self.logger.info(f"산업별 피벗 완료: {result_df.shape}")
        return result_df
    
    def aggregate_subdistricts_to_city(self, df: pd.DataFrame) -> pd.DataFrame:
        """시군구가 매핑과 불일치하는 하위 구 단위를 같은 시 단위로 합산합니다.
        기준 매핑: data/processed/시도_시군구_매핑.csv
        합산: 제조업_종사자수, 서비스업_종사자수 합계
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
        
        # 제주도 특별 처리: 제주시, 서귀포시를 그대로 유지
        jeju_mask = work['시도'] == '제주특별자치도'
        work.loc[jeju_mask, '표준시군구'] = work.loc[jeju_mask, '시군구']
        
        grouped = (
            work.groupby(['시도','표준시군구','연도','반기'], as_index=False)[['제조업_종사자수','서비스업_종사자수']]
                .sum()
        )
        grouped = grouped.rename(columns={'표준시군구':'시군구'})
        grouped['제조업_종사자수'] = grouped['제조업_종사자수'].fillna(0).round().astype(int)
        grouped['서비스업_종사자수'] = grouped['서비스업_종사자수'].fillna(0).round().astype(int)
        return grouped

    def process_data(self, start_year: int = 2019, end_year: int = 2023) -> pd.DataFrame:
        """전체 데이터 전처리 파이프라인을 실행합니다."""
        try:
            self.logger.info("산업별 종사자 데이터 전처리 시작")
            
            # 전체 파이프라인 실행
            result_df = self.pivot_industry_data()
            # 시 하위 구 합산 집계
            result_df = self.aggregate_subdistricts_to_city(result_df)
            
            self.logger.info(f"산업별 종사자 데이터 전처리 완료: {result_df.shape}")
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
            output_path = os.path.join(base_dir, "data", "processed", f"산업별_종사자_전처리완료_{encoding}.csv")
        
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
    industry_processor = IndustryProcessor()
    
    try:
        # 데이터 전처리 실행
        processed_data = industry_processor.process_data()
        
        # 요약 정보 출력
        # summary = industry_processor.get_summary() # get_summary 메서드는 삭제되었으므로 주석 처리
        # print("=== 산업별 종사자 데이터 전처리 결과 ===")
        # for key, value in summary.items():
        #     print(f"{key}: {value}")
        
        # 데이터 미리보기
        print(f"\n=== 데이터 미리보기 ===")
        print(processed_data.head())
        
        # 결과 내보내기
        industry_processor.export_both_encodings(processed_data)
        
    except Exception as e:
        print(f"전처리 실패: {e}")
