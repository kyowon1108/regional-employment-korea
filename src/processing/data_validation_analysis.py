#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
원본 데이터와 최종 통합 데이터 비교 분석 스크립트
"""

import pandas as pd
import numpy as np
import os
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

def analyze_final_data():
    """최종 통합 데이터를 분석합니다."""
    print("=== 최종 통합 데이터 분석 ===")
    
    final_data_path = DATA_PROCESSED_DIR / "최종_통합데이터_수정_cp949.csv"
    final_df = load_csv_with_encoding(final_data_path)
    
    if final_df is None:
        return
    
    print(f"총 행 수: {len(final_df)}")
    print(f"총 열 수: {len(final_df.columns)}")
    print("\n컬럼명:")
    for i, col in enumerate(final_df.columns):
        print(f"{i+1}. {col}")
    
    print("\n데이터 샘플 (처음 5행):")
    print(final_df.head())
    
    print("\n데이터 타입:")
    print(final_df.dtypes)
    
    print("\n기본 통계:")
    print(final_df.describe())
    
    return final_df

def analyze_original_data():
    """원본 데이터들을 분석합니다."""
    print("\n=== 원본 데이터 분석 ===")
    
    original_files = list(DATA_RAW_DIR.glob("*.csv"))
    
    for file_path in original_files:
        print(f"\n--- {file_path.name} 분석 ---")
        df = load_csv_with_encoding(file_path)
        
        if df is None:
            continue
            
        print(f"크기: {df.shape}")
        print(f"컬럼: {list(df.columns)}")
        print("처음 3행:")
        print(df.head(3))

def compare_data_sources():
    """원본 데이터와 최종 데이터를 비교합니다."""
    print("\n=== 데이터 소스 비교 분석 ===")
    
    # 최종 데이터 로드
    final_df = load_csv_with_encoding(DATA_PROCESSED_DIR / "최종_통합데이터_수정_cp949.csv")
    if final_df is None:
        return
    
    # E9 체류자 데이터 로드
    e9_df = load_csv_with_encoding(DATA_RAW_DIR / "시군구별_E9_체류자(2014~2023).csv")
    if e9_df is None:
        return
    
    # 산업별 고용 데이터 로드
    industry_df = load_csv_with_encoding(DATA_RAW_DIR / "산업별 고용 시군구 2025-08-22.csv")
    if industry_df is None:
        return
    
    print("1. E9 체류자 데이터 비교:")
    print(f"   원본 E9 데이터 행 수: {len(e9_df)}")
    print(f"   최종 데이터에서 E9 관련 행 수: {len(final_df[final_df.iloc[:, 4] > 0])}")
    
    print("\n2. 산업별 고용 데이터 비교:")
    print(f"   원본 산업별 고용 데이터 행 수: {len(industry_df)}")
    
    # 지역별 데이터 수 비교
    final_regions = final_df.iloc[:, 0].unique()
    e9_regions = e9_df.iloc[:, 0].unique()
    
    print(f"\n3. 지역 수 비교:")
    print(f"   최종 데이터 지역 수: {len(final_regions)}")
    print(f"   E9 데이터 지역 수: {len(e9_regions)}")
    
    # 공통 지역 확인
    common_regions = set(final_regions) & set(e9_regions)
    print(f"   공통 지역 수: {len(common_regions)}")
    
    print("\n4. 연도 범위 비교:")
    final_years = final_df.iloc[:, 3].unique()
    print(f"   최종 데이터 연도: {sorted(final_years)}")
    
    if e9_df.shape[1] > 3:
        e9_years = e9_df.columns[3:].tolist()
        print(f"   E9 데이터 연도: {e9_years}")

def validate_data_consistency():
    """데이터 일관성을 검증합니다."""
    print("\n=== 데이터 일관성 검증 ===")
    
    final_df = load_csv_with_encoding(DATA_PROCESSED_DIR / "최종_통합데이터_수정_cp949.csv")
    if final_df is None:
        return
    
    # 1. 결측값 확인
    print("1. 결측값 확인:")
    missing_data = final_df.isnull().sum()
    if missing_data.sum() > 0:
        print("   결측값이 발견되었습니다:")
        print(missing_data[missing_data > 0])
    else:
        print("   결측값이 없습니다.")
    
    # 2. 데이터 타입 검증
    print("\n2. 데이터 타입 검증:")
    print(final_df.dtypes)
    
    # 3. 음수값 확인
    print("\n3. 음수값 확인:")
    numeric_cols = final_df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        negative_count = (final_df[col] < 0).sum()
        if negative_count > 0:
            print(f"   {col}: {negative_count}개의 음수값")
    
    # 4. 연도 범위 검증
    print("\n4. 연도 범위 검증:")
    years = final_df.iloc[:, 3].unique()
    print(f"   연도 범위: {min(years)} ~ {max(years)}")
    
    # 5. 계절 구분 검증
    print("\n5. 계절 구분 검증:")
    seasons = final_df.iloc[:, 4].unique()
    print(f"   계절 구분: {sorted(seasons)}")

def generate_validation_report():
    """검증 보고서를 생성합니다."""
    print("\n" + "="*60)
    print("데이터 검증 분석 보고서")
    print("="*60)
    
    # 각 분석 함수 실행
    final_df = analyze_final_data()
    analyze_original_data()
    compare_data_sources()
    validate_data_consistency()
    
    print("\n" + "="*60)
    print("검증 완료")
    print("="*60)

if __name__ == "__main__":
    generate_validation_report()
