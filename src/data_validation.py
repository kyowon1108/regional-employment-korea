import pandas as pd
import numpy as np

def validate_preprocessed_data():
    """전처리된 데이터 검증"""
    print("=== 전처리된 데이터 검증 ===\\n")

    # 데이터 로드
    df = pd.read_csv("/Users/kapr/Desktop/DataAnalyze/new_analysis/data/new_processed/comprehensive_integrated_data.csv")
    summary_df = pd.read_csv("/Users/kapr/Desktop/DataAnalyze/new_analysis/data/new_processed/comprehensive_summary.csv")

    print(f"📊 패널 데이터: {df.shape}")
    print(f"📋 요약 데이터: {summary_df.shape}")

    # 1. 기본 현황
    print("\\n1. 기본 현황")
    print(f"   • 지자체 수: {df[['시도', '시군구']].drop_duplicates().shape[0]}개")
    print(f"   • 분석 기간: {df['연도'].min()}-{df['연도'].max()}년")
    print(f"   • 시도 수: {df['시도'].nunique()}개")

    # 2. 시도별 분포
    print("\\n2. 시도별 지자체 분포")
    sido_counts = df.groupby('시도')['시군구'].nunique().sort_values(ascending=False)
    for sido, count in sido_counts.items():
        sido_short = sido.replace('특별자치도', '').replace('광역시', '').replace('특별시', '').replace('도', '')
        print(f"   • {sido_short}: {count}개")

    # 3. E9 체류자 현황
    print("\\n3. E9 체류자 현황 (5년 평균 기준)")
    e9_stats = summary_df['E9_체류자수']
    print(f"   • 평균: {e9_stats.mean():.0f}명")
    print(f"   • 최대: {e9_stats.max():.0f}명")
    print(f"   • E9 > 0인 지역: {len(summary_df[summary_df['E9_체류자수'] > 0])}개")

    # E9 상위 10개 지역
    print("\\n   📍 E9 체류자수 상위 10개 지자체:")
    top10_e9 = summary_df.nlargest(10, 'E9_체류자수')[['시도', '시군구', 'E9_체류자수', '고용률']]
    for i, (_, row) in enumerate(top10_e9.iterrows(), 1):
        sido_short = row['시도'].replace('특별자치도', '').replace('광역시', '').replace('특별시', '').replace('도', '')
        print(f"   {i:2d}. {sido_short} {row['시군구']:8s}: {row['E9_체류자수']:6.0f}명 (고용률: {row['고용률']:.1f}%)")

    # 4. 고용률 현황
    print("\\n4. 고용률 현황 (5년 평균 기준)")
    emp_stats = summary_df['고용률']
    print(f"   • 평균: {emp_stats.mean():.1f}%")
    print(f"   • 최고: {emp_stats.max():.1f}% ({summary_df.loc[summary_df['고용률'].idxmax(), '시군구']})")
    print(f"   • 최저: {emp_stats.min():.1f}% ({summary_df.loc[summary_df['고용률'].idxmin(), '시군구']})")

    # 고용률 상위 10개 지역
    print("\\n   📍 고용률 상위 10개 지자체:")
    top10_emp = summary_df.nlargest(10, '고용률')[['시도', '시군구', '고용률', 'E9_체류자수']]
    for i, (_, row) in enumerate(top10_emp.iterrows(), 1):
        sido_short = row['시도'].replace('특별자치도', '').replace('광역시', '').replace('특별시', '').replace('도', '')
        print(f"   {i:2d}. {sido_short} {row['시군구']:8s}: {row['고용률']:5.1f}% (E9: {row['E9_체류자수']:4.0f}명)")

    # 5. 상관관계 확인
    print("\\n5. 주요 변수 간 상관관계")
    corr_vars = ['고용률', 'E9_체류자수', '제조업_비중', '서비스업_비중']
    corr_matrix = summary_df[corr_vars].corr()

    print(f"   • 고용률 ⟷ E9 체류자수: {corr_matrix.loc['고용률', 'E9_체류자수']:.3f}")
    print(f"   • 고용률 ⟷ 제조업비중:  {corr_matrix.loc['고용률', '제조업_비중']:.3f}")
    print(f"   • 고용률 ⟷ 서비스업비중: {corr_matrix.loc['고용률', '서비스업_비중']:.3f}")

    # 6. 연도별 추이
    print("\\n6. 연도별 평균 추이")
    yearly_stats = df.groupby('연도').agg({
        '고용률': 'mean',
        'E9_체류자수': 'mean',
        '제조업_비중': 'mean'
    })

    print("   연도    고용률    E9평균   제조업비중")
    print("   " + "-"*35)
    for year, row in yearly_stats.iterrows():
        print(f"   {year}   {row['고용률']:5.1f}%   {row['E9_체류자수']:6.0f}명   {row['제조업_비중']:5.1f}%")

    # 7. 데이터 완전성 확인
    print("\\n7. 데이터 완전성 확인")

    # 각 지자체별 연도 수 확인
    completeness = df.groupby(['시도', '시군구']).size()
    incomplete = completeness[completeness != 5]  # 5년이 아닌 지역

    if len(incomplete) > 0:
        print(f"   ⚠️ 불완전한 데이터 지역: {len(incomplete)}개")
        for (sido, sigungu), count in incomplete.head().items():
            print(f"      - {sido} {sigungu}: {count}년")
    else:
        print("   ✅ 모든 지자체가 5년 완전 데이터 보유")

    # 8. 결측값 확인
    print("\\n8. 결측값 현황")
    missing_info = df.isnull().sum()
    total_missing = missing_info.sum()

    if total_missing == 0:
        print("   ✅ 결측값 없음")
    else:
        print(f"   ⚠️ 총 {total_missing}개 결측값:")
        for col, count in missing_info.items():
            if count > 0:
                print(f"      - {col}: {count}개")

    print("\\n" + "="*50)
    print("📋 전처리 완료 상태: ✅ 양호")
    print(f"📊 최종 분석 준비 데이터: {len(summary_df)}개 지자체 × 5년 = {len(df)}개 관측치")
    print("="*50)

    return df, summary_df

def identify_potential_issues():
    """잠재적 문제점 식별"""
    print("\\n=== 잠재적 이슈 확인 ===")

    df = pd.read_csv("/Users/kapr/Desktop/DataAnalyze/new_analysis/data/new_processed/comprehensive_integrated_data.csv")
    summary_df = pd.read_csv("/Users/kapr/Desktop/DataAnalyze/new_analysis/data/new_processed/comprehensive_summary.csv")

    issues = []

    # 1. 극값 확인
    emp_q1, emp_q3 = summary_df['고용률'].quantile([0.25, 0.75])
    emp_iqr = emp_q3 - emp_q1
    emp_outliers = summary_df[
        (summary_df['고용률'] < emp_q1 - 1.5*emp_iqr) |
        (summary_df['고용률'] > emp_q3 + 1.5*emp_iqr)
    ]

    if len(emp_outliers) > 0:
        issues.append(f"고용률 극값: {len(emp_outliers)}개 지역")
        print(f"🔍 고용률 극값 지역 ({len(emp_outliers)}개):")
        for _, row in emp_outliers[['시군구', '고용률']].iterrows():
            print(f"   • {row['시군구']}: {row['고용률']:.1f}%")

    # 2. E9 집중도 확인
    e9_total = summary_df['E9_체류자수'].sum()
    if e9_total > 0:
        top5_e9_sum = summary_df.nlargest(5, 'E9_체류자수')['E9_체류자수'].sum()
        concentration = top5_e9_sum / e9_total * 100

        if concentration > 80:
            issues.append(f"E9 집중도 높음: 상위 5개 지역이 {concentration:.1f}% 점유")
            print(f"🔍 E9 집중도: 상위 5개 지역이 전체의 {concentration:.1f}% 점유")

    # 3. 0값 비율 확인
    zero_e9 = len(summary_df[summary_df['E9_체류자수'] == 0])
    zero_ratio = zero_e9 / len(summary_df) * 100

    if zero_ratio > 70:
        issues.append(f"E9 제로 지역 많음: {zero_ratio:.1f}%")
        print(f"🔍 E9 체류자 0명 지역: {zero_e9}개 ({zero_ratio:.1f}%)")

    # 4. 제조업 비중 확인
    high_manufacturing = len(summary_df[summary_df['제조업_비중'] > 60])
    if high_manufacturing > 0:
        print(f"🔍 제조업 고집중 지역: {high_manufacturing}개 (60% 이상)")

    if len(issues) == 0:
        print("✅ 특별한 이슈 없음")
    else:
        print(f"\\n⚠️ 확인된 이슈: {len(issues)}개")
        for issue in issues:
            print(f"   • {issue}")

    print("\\n💡 추천 사항:")
    print("   1. E9 집중 현상으로 인해 상관관계 분석 시 주의 필요")
    print("   2. 극값 지역에 대한 별도 분석 고려")
    print("   3. 제조업/서비스업 비중을 활용한 산업구조 분석 가능")

def main():
    """메인 검증 함수"""
    df, summary_df = validate_preprocessed_data()
    identify_potential_issues()

    return df, summary_df

if __name__ == "__main__":
    df, summary_df = main()