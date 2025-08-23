#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
설정 관리 클래스
"""

from pathlib import Path
from dataclasses import dataclass

@dataclass
class Config:
    """데이터 처리 설정"""
    
    # 데이터 폴더 경로
    data_dir: Path = Path("data")
    processed_dir: Path = Path("data/processed")
    
    # 입력 파일 경로
    area_file: Path = Path("data/raw/지역_면적_utf8.csv")
    e9_file: Path = Path("data/raw/시군구별_E9_체류자(2014~2023).csv")
    industry_file: Path = Path("data/raw/지역별_산업별_종사자.csv")
    employment_file: Path = Path("data/raw/시군구별_경제활동인구_(2021~2024반기).csv")
    
    # 출력 파일 경로
    output_utf8: Path = Path("data/processed/최종.csv")
    output_cp949: Path = Path("data/processed/최종_cp949.csv")
    
    # 로그 폴더
    log_dir: Path = Path("logs")
    
    def __post_init__(self):
        """초기화 후 검증"""
        # 필요한 폴더 생성
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(exist_ok=True)
        
        # 입력 파일 존재 확인
        required_files = [
            self.area_file,
            self.e9_file,
            self.industry_file,
            self.employment_file
        ]
        
        for file_path in required_files:
            if not file_path.exists():
                raise FileNotFoundError(f"필수 파일이 없습니다: {file_path}")
    
    def get_data_paths(self):
        """데이터 경로 정보 반환"""
        return {
            'area': self.area_file,
            'e9': self.e9_file,
            'industry': self.industry_file,
            'employment': self.employment_file
        }
    
    def get_output_paths(self):
        """출력 경로 정보 반환"""
        return {
            'utf8': self.output_utf8,
            'cp949': self.output_cp949
        }
