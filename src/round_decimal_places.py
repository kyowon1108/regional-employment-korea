import pandas as pd
import numpy as np

def round_comprehensive_data():
    """comprehensive_integrated_data.csv의 소수점을 2자리로 반올림"""
    print("=== 소수점 반올림 처리 시작 ===")

    # 데이터 로드
    file_path = "/Users/kapr/Desktop/DataAnalyze/new_analysis/data/new_processed/comprehensive_integrated_data.csv"
    df = pd.read_csv(file_path)

    print(f"처리 전 데이터: {df.shape}")

    # 수치형 컬럼 확인
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    print(f"수치형 컬럼: {numeric_columns}")

    # 소수점 2자리로 반올림
    for col in numeric_columns:
        df[col] = df[col].round(2)

    # 저장
    df.to_csv(file_path, index=False)

    print("✅ 소수점 반올림 완료 - comprehensive_integrated_data.csv 업데이트됨")

    # 결과 확인
    print("\n처리 후 몇 개 행 확인:")
    print(df.head(3))

    return df

def round_summary_data():
    """comprehensive_summary.csv도 함께 반올림"""
    print("\n=== 요약 데이터 소수점 반올림 ===")

    file_path = "/Users/kapr/Desktop/DataAnalyze/new_analysis/data/new_processed/comprehensive_summary.csv"
    df = pd.read_csv(file_path)

    # 수치형 컬럼 반올림
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_columns:
        df[col] = df[col].round(2)

    # 저장
    df.to_csv(file_path, index=False)

    print("✅ 요약 데이터 소수점 반올림 완료")

    return df

if __name__ == "__main__":
    # 두 파일 모두 처리
    df_integrated = round_comprehensive_data()
    df_summary = round_summary_data()

    print("\n=== 모든 반올림 처리 완료 ===")