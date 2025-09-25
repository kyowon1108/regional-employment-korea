#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
상세 데이터 검증 및 비교 분석 스크립트
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Set up paths - updated for new folder structure
ROOT_DIR = Path(__file__).parent.parent.parent
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
DATA_RAW_DIR = ROOT_DIR / "data" / "raw"

def load_csv_with_encoding(file_path, encoding='cp949'):
    """인코딩을 시도하여 CSV 파일을 로드합니다."""
    try:
        return pd.read_csv(file_path, encoding=encoding)
    except UnicodeDecodeError:
        try:
            return pd.read_csv(file_path, encoding='utf-8')
        except UnicodeDecodeError:
            print(f"파일을 읽을 수 없습니다: {file_path}")
            return None

def detailed_e9_comparison():
    """E9 체류자 데이터 상세 비교"""
    print("=== E9 체류자 데이터 상세 비교 ===")
    
    # 최종 데이터 로드
    final_df = load_csv_with_encoding(DATA_PROCESSED_DIR / "최종_통합데이터_수정_cp949.csv")
    e9_df = load_csv_with_encoding(DATA_RAW_DIR / "시군구별_E9_체류자(2014~2023).csv")
    
    if final_df is None or e9_df is None:
        return
    
    # E9 데이터에서 2019-2023년 데이터만 추출
    e9_recent = e9_df[['행정구역(시군구)별(1)', '2019', '2020', '2021', '2022', '2023']].copy()
    
    # 성별이 '계'인 데이터만 사용
    e9_recent = e9_recent[e9_df['성별(1)'] == '계'].copy()
    
    print(f"E9 원본 데이터 (2019-2023): {len(e9_recent)} 행")
    
    # 최종 데이터에서 E9 값이 있는 데이터 추출
    final_e9_data = final_df[final_df['E9_체류자수'] > 0].copy()
    print(f"최종 데이터에서 E9 값이 있는 행: {len(final_e9_data)} 행")
    
    # 지역별 비교
    final_regions = set(final_df['시도'].unique())
    e9_regions = set(e9_recent['행정구역(시군구)별(1)'].unique())
    
    print(f"\n지역 비교:")
    print(f"최종 데이터 지역: {sorted(final_regions)}")
    print(f"E9 데이터 지역: {sorted(e9_regions)}")
    
    # 공통 지역 확인
    common_regions = final_regions & e9_regions
    print(f"공통 지역: {sorted(common_regions)}")
    
    # 특정 지역의 E9 데이터 비교 (예: 서울특별시)
    if '서울특별시' in common_regions:
        print(f"\n서울특별시 E9 데이터 비교:")
        
        # 원본 E9 데이터
        e9_seoul = e9_recent[e9_recent['행정구역(시군구)별(1)'] == '서울특별시']
        print(f"원본 E9 서울 데이터:")
        print(e9_seoul[['2019', '2020', '2021', '2022', '2023']].values)
        
        # 최종 데이터
        final_seoul = final_df[final_df['시도'] == '서울특별시']
        seoul_e9_by_year = final_seoul.groupby('연도')['E9_체류자수'].sum()
        print(f"최종 데이터 서울 E9 합계:")
        print(seoul_e9_by_year)

def detailed_industry_comparison():
    """산업별 고용 데이터 상세 비교"""
    print("\n=== 산업별 고용 데이터 상세 비교 ===")
    
    final_df = load_csv_with_encoding(DATA_PROCESSED_DIR / "최종_통합데이터_수정_cp949.csv")
    industry_df = load_csv_with_encoding(DATA_RAW_DIR / "산업별 고용 시군구 2025-08-22.csv")
    
    if final_df is None or industry_df is None:
        return
    
    print(f"산업별 고용 원본 데이터: {industry_df.shape}")
    
    # 원본 데이터에서 제조업과 서비스업 데이터 추출
    manufacturing_data = industry_df[industry_df['산업별'] == '광업.제조업(B,C)'].copy()
    service_data = industry_df[industry_df['산업별'] == '사업ㆍ개인ㆍ공공서비스업(E,L~S)'].copy()
    
    print(f"제조업 데이터: {len(manufacturing_data)} 행")
    print(f"서비스업 데이터: {len(service_data)} 행")
    
    # 지역별 비교
    final_regions = set(final_df['시도'].unique())
    industry_regions = set(manufacturing_data['지역별'].unique())
    
    print(f"\n지역 비교:")
    print(f"최종 데이터 지역: {sorted(final_regions)}")
    print(f"산업별 고용 데이터 지역: {sorted(industry_regions)}")
    
    # 특정 지역의 데이터 비교 (예: 서울특별시 종로구)
    if '서울특별시 종로구' in industry_regions:
        print(f"\n서울특별시 종로구 데이터 비교:")
        
        # 원본 제조업 데이터
        orig_manufacturing = manufacturing_data[manufacturing_data['지역별'] == '서울특별시 종로구']
        print(f"원본 제조업 데이터 (2021년):")
        print(f"2021 1/2: {orig_manufacturing['2021 1/2'].iloc[0]}")
        print(f"2021 2/2: {orig_manufacturing['2021 2/2'].iloc[0]}")
        
        # 원본 서비스업 데이터
        orig_service = service_data[service_data['지역별'] == '서울특별시 종로구']
        print(f"원본 서비스업 데이터 (2021년):")
        print(f"2021 1/2: {orig_service['2021 1/2'].iloc[0]}")
        print(f"2021 2/2: {orig_service['2021 2/2'].iloc[0]}")
        
        # 최종 데이터
        final_jongno = final_df[(final_df['시도'] == '서울특별시') & (final_df['시군구'] == '종로구')]
        print(f"최종 데이터:")
        for _, row in final_jongno.iterrows():
            print(f"{row['연도']}년 {row['반기']}반기: 제조업 {row['제조업_종사자수']}, 서비스업 {row['서비스업_종사자수']}")

def employment_rate_analysis():
    """고용률 데이터 분석"""
    print("\n=== 고용률 데이터 분석 ===")
    
    final_df = load_csv_with_encoding(DATA_PROCESSED_DIR / "최종_통합데이터_수정_cp949.csv")
    employment_df = load_csv_with_encoding(DATA_RAW_DIR / "시군구_연령별_취업자_및_고용률.csv")
    
    if final_df is None or employment_df is None:
        return
    
    print(f"고용률 원본 데이터: {employment_df.shape}")
    
    # 원본 데이터에서 15-64세 고용률 데이터 추출
    employment_15_64 = employment_df[
        (employment_df['연령별'] == '15 - 64세') & 
        (employment_df['항목'] == '고용률 (%)')
    ].copy()
    
    print(f"15-64세 고용률 데이터: {len(employment_15_64)} 행")
    
    # 지역별 비교
    final_regions = set(final_df['시도'].unique())
    employment_regions = set(employment_15_64['행정구역별'].str.split().str[0].unique())
    
    print(f"\n지역 비교:")
    print(f"최종 데이터 지역: {sorted(final_regions)}")
    print(f"고용률 데이터 지역: {sorted(employment_regions)}")
    
    # 특정 지역의 고용률 비교 (예: 서울)
    if '서울' in employment_regions:
        print(f"\n서울 고용률 데이터 비교:")
        
        # 원본 데이터
        seoul_employment = employment_15_64[employment_15_64['행정구역별'].str.startswith('서울')]
        print(f"원본 서울 고용률 데이터:")
        for _, row in seoul_employment.iterrows():
            print(f"{row['행정구역별']}: 2021.1.2={row['2021.1.2']}, 2021.2.2={row['2021.2.2']}")
        
        # 최종 데이터
        final_seoul = final_df[final_df['시도'] == '서울특별시']
        seoul_employment_final = final_seoul.groupby(['시군구', '연도'])['고용률'].mean()
        print(f"\n최종 데이터 서울 고용률:")
        print(seoul_employment_final)

def data_quality_check():
    """데이터 품질 검사"""
    print("\n=== 데이터 품질 검사 ===")
    
    final_df = load_csv_with_encoding(DATA_PROCESSED_DIR / "최종_통합데이터_수정_cp949.csv")
    
    if final_df is None:
        return
    
    print("1. 데이터 완성도 검사:")
    
    # 연도별 데이터 수
    year_counts = final_df['연도'].value_counts().sort_index()
    print(f"연도별 데이터 수:")
    for year, count in year_counts.items():
        print(f"  {year}년: {count}행")
    
    # 지역별 데이터 수
    region_counts = final_df['시도'].value_counts()
    print(f"\n지역별 데이터 수:")
    for region, count in region_counts.items():
        print(f"  {region}: {count}행")
    
    print("\n2. 데이터 일관성 검사:")
    
    # 면적 데이터 검사
    area_consistency = final_df.groupby(['시도', '시군구'])['면적'].nunique()
    inconsistent_areas = area_consistency[area_consistency > 1]
    if len(inconsistent_areas) > 0:
        print(f"면적이 일관되지 않은 지역: {len(inconsistent_areas)}개")
        print(inconsistent_areas)
    else:
        print("면적 데이터가 일관됩니다.")
    
    # E9 체류자수와 취업자 관계 검사
    print("\n3. E9 체류자수와 취업자 관계:")
    e9_employment_relation = final_df[final_df['E9_체류자수'] > 0]
    print(f"E9 체류자수가 있는 데이터: {len(e9_employment_relation)}행")
    
    # 취업자가 0인 경우
    zero_employment = e9_employment_relation[e9_employment_relation['취업자'] == 0]
    print(f"E9 체류자수가 있지만 취업자가 0인 데이터: {len(zero_employment)}행")
    
    # 고용률이 0인 경우
    zero_rate = e9_employment_relation[e9_employment_relation['고용률'] == 0]
    print(f"E9 체류자수가 있지만 고용률이 0인 데이터: {len(zero_rate)}행")

def generate_detailed_report():
    """상세 검증 보고서 생성"""
    print("="*80)
    print("상세 데이터 검증 분석 보고서")
    print("="*80)
    
    detailed_e9_comparison()
    detailed_industry_comparison()
    employment_rate_analysis()
    data_quality_check()
    
    print("\n" + "="*80)
    print("상세 검증 완료")
    print("="*80)

if __name__ == "__main__":
    generate_detailed_report()
