import pandas as pd

def check_region_matching():
    """지역명 매칭 상태 확인"""
    print("=== 지역명 매칭 상태 확인 ===")

    # 1. 표준 매핑 테이블
    mapping_df = pd.read_csv("/Users/kapr/Desktop/DataAnalyze/data/processed/시도_시군구_매핑.csv")
    print(f"표준 매핑 테이블: {len(mapping_df)}개 지자체")

    # 2. 전처리된 최종 데이터
    final_df = pd.read_csv("/Users/kapr/Desktop/DataAnalyze/new_analysis/data/new_processed/comprehensive_integrated_data.csv")
    final_regions = set(final_df['시군구'].unique())
    print(f"최종 데이터 지자체: {len(final_regions)}개")

    # 3. 매핑 테이블과 비교
    mapping_regions = set(mapping_df['시군구'].unique())

    # 매핑 테이블에는 있지만 최종 데이터에는 없는 지역
    missing_in_final = mapping_regions - final_regions
    # 최종 데이터에는 있지만 매핑 테이블에는 없는 지역
    missing_in_mapping = final_regions - mapping_regions

    print(f"\\n📍 매핑 테이블에는 있지만 최종 데이터에 없는 지역: {len(missing_in_final)}개")
    if len(missing_in_final) > 0:
        for i, region in enumerate(sorted(missing_in_final)[:20], 1):
            print(f"   {i:2d}. {region}")
        if len(missing_in_final) > 20:
            print(f"   ... 외 {len(missing_in_final)-20}개")

    print(f"\\n📍 최종 데이터에는 있지만 매핑 테이블에 없는 지역: {len(missing_in_mapping)}개")
    if len(missing_in_mapping) > 0:
        for i, region in enumerate(sorted(missing_in_mapping), 1):
            print(f"   {i:2d}. {region}")

    # 4. 원본 고용률 데이터의 지역명과 비교
    employment_raw = pd.read_csv("/Users/kapr/Desktop/DataAnalyze/new_analysis/data/new_processed/employment_raw.csv")

    print(f"\\n📍 원본 고용률 데이터 지역 확인:")
    original_regions = employment_raw['행정구역'].unique()
    print(f"원본 지역 수: {len(original_regions)}개")

    # 처리 과정에서 제외된 지역들 확인 ('-' 값으로 인해)
    print("\\n📍 처리 과정에서 제외된 지역들 (2019-2023년 완전 데이터 없음):")

    excluded_regions = []
    for region in original_regions:
        # 해당 지역의 2019-2023년 데이터 확인
        row = employment_raw[employment_raw['행정구역'] == region].iloc[0]

        years_data = [
            row['2019.1/2'], row['2019.2/2'],
            row['2020.1/2'], row['2020.2/2'],
            row['2021.1/2'], row['2021.2/2'],
            row['2022.1/2'], row['2022.2/2'],
            row['2023.1/2'], row['2023.2/2']
        ]

        # '-' 값이 있으면 제외된 지역
        if '-' in years_data:
            excluded_regions.append(region)

    print(f"제외된 지역 수: {len(excluded_regions)}개")
    if len(excluded_regions) > 0:
        for i, region in enumerate(sorted(excluded_regions)[:15], 1):
            print(f"   {i:2d}. {region}")
        if len(excluded_regions) > 15:
            print(f"   ... 외 {len(excluded_regions)-15}개")

    # 5. 특이한 지역명 패턴 확인
    print(f"\\n📍 최종 데이터 지역명 패턴 확인:")

    # 시도별 지역 분포
    final_sido_dist = final_df.groupby('시도')['시군구'].nunique().sort_values(ascending=False)
    print("시도별 지자체 수:")
    for sido, count in final_sido_dist.items():
        sido_short = sido.replace('특별자치도', '').replace('광역시', '').replace('특별시', '').replace('도', '')
        print(f"   • {sido_short}: {count}개")

    return {
        'total_mapping': len(mapping_df),
        'total_final': len(final_regions),
        'missing_in_final': missing_in_final,
        'missing_in_mapping': missing_in_mapping,
        'excluded_regions': excluded_regions
    }

def suggest_additional_processing():
    """추가 전처리 필요사항 제안"""
    print("\\n=== 추가 전처리 필요사항 검토 ===")

    result = check_region_matching()

    recommendations = []

    # 1. 누락된 지역이 있는 경우
    if len(result['missing_in_mapping']) > 0:
        recommendations.append("🔧 매핑 테이블에 없는 지역명 수동 매핑 필요")

    # 2. 많은 지역이 제외된 경우
    if len(result['excluded_regions']) > 50:
        recommendations.append("🔧 2019-2020년 데이터 부족으로 많은 지역 제외됨")

    # 3. 데이터 품질 향상 방안
    recommendations.extend([
        "🔧 제조업 고집중 지역(13개) 별도 분석 고려",
        "🔧 E9-고용률 음의 상관관계(-0.196) 심층 분석 필요",
        "🔧 울릉군 등 극값 지역 이상치 처리 검토"
    ])

    print("\\n💡 추가 처리 권장사항:")
    for i, rec in enumerate(recommendations, 1):
        print(f"   {i}. {rec}")

    # 4. 현재 전처리 상태 평가
    print("\\n📋 전처리 상태 종합 평가:")

    score = 100
    issues = []

    if len(result['missing_in_mapping']) > 0:
        score -= 10
        issues.append(f"매핑되지 않은 지역 {len(result['missing_in_mapping'])}개")

    if len(result['excluded_regions']) > 100:
        score -= 15
        issues.append(f"제외된 지역이 많음 ({len(result['excluded_regions'])}개)")

    if score >= 90:
        status = "🟢 우수"
    elif score >= 80:
        status = "🟡 양호"
    else:
        status = "🔴 보완 필요"

    print(f"   전처리 점수: {score}/100점 {status}")
    if issues:
        print(f"   개선 사항: {', '.join(issues)}")
    else:
        print("   ✅ 특별한 개선사항 없음")

    print("\\n🎯 결론:")
    if score >= 85:
        print("   현재 전처리 상태가 분석에 충분히 적합합니다.")
        print("   바로 데이터 분석을 진행하셔도 됩니다.")
    else:
        print("   몇 가지 보완이 필요하지만 기본적인 분석은 가능합니다.")

    return score

def main():
    result = check_region_matching()
    score = suggest_additional_processing()

    return result, score

if __name__ == "__main__":
    result, score = main()