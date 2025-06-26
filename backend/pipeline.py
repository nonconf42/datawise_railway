from typing import Tuple, Dict, List
from dataclasses import dataclass, field
from DCAgent import DataCleaningAgent
from datasets.data import DataReader
from llms.llm import LLM
from pprint import pprint
from data_aggregation_agent import DataAggregationAgent

@dataclass
class LLMConfig:
    """Configuration for LLM models"""
    platform: str
    model_name: str

def default_llm_config() -> LLMConfig:
    """Default LLM config for data aggregation"""
    return LLMConfig(
        platform='anthropic',
        model_name="claude-3-7-sonnet-20250219"
    )

@dataclass
class PipelineConfig:
    """Configuration for the data analysis pipeline"""
    # Data cleaning config
    dc_missing_threshold: float = 1
    dc_outlier_method: str = 'iqr'
    dc_outlier_threshold: float = 1000
    dc_handle_categorical: bool = True
    # Data aggregation config
    da_num_recommendations: int = 5
    da_iterations: int = 3  # Number of times to run the aggregation generator
    llm_config: LLMConfig = field(default_factory=default_llm_config)

class DataAnalysisPipeline:
    def __init__(self, dataset, config: PipelineConfig):
        """
        Initialize the data analysis pipeline.
        
        Args:
            config: Pipeline configuration
        """
        self.config = config
        self.dataset = dataset
        
        # Initialize data cleaning agent
        self.dc_agent = DataCleaningAgent(
            missing_threshold=config.dc_missing_threshold,
            outlier_method=config.dc_outlier_method,
            outlier_threshold=config.dc_outlier_threshold,
            handle_categorical=config.dc_handle_categorical
        )
        
        # Initialize LLM
        self.llm = LLM(config.llm_config.platform)
        
        # Initialize Data Aggregation Agent
        self.da_agent = DataAggregationAgent(
            llm_model=self.llm,
            dataset=self.dataset,
            num_recommendations=config.da_num_recommendations,
            # iterations=config.da_iterations
        )

    def run(self):
        """
        Run the data analysis pipeline with only data cleaning and aggregation.
        
        Returns:
            List of data aggregations with tables and analyses
        """
        print("Starting Data Analysis Pipeline...")
        
        # Step 1: Clean the data
        print("\nStep 1: Data Cleaning")
        clean_df = self.dc_agent.clean(self.dataset.train_base)
        cleaning_stats = self.dc_agent.get_cleaning_stats()
        print("Data cleaning completed. Stats:", cleaning_stats)
        print("Data columns:", self.dataset.train_base.columns)
        
        # Update dataset with cleaned data
        self.dataset.train_base = clean_df
        self.dataset.train_base.columns = [col.strip() for col in self.dataset.train_base.columns]
        print("Data columns after strip:", self.dataset.train_base.columns)
        self.dataset.features_description = {k: v for k, v in self.dataset.features_description.items() if k in self.dataset.train_base.columns}
        pprint(self.dataset.features_description)
        
        # Step 2: Run Data Aggregation Agent
        print("\nStep 2: Running Data Aggregation Agent")
        aggregations = self.da_agent.run()
        
        return aggregations

def run_pipeline(
    dataset,
    llm_model: LLMConfig = None
):
    """
    Run the data analysis pipeline for a given dataset.
    
    Args:
        dataset: Dataset object to analyze
        llm_model: Optional custom LLM config
        
    Returns:
        List of data aggregations with tables and analyses
    """
    config_params = {
        'dc_missing_threshold': 1,
        'dc_outlier_method': 'iqr',
        'dc_outlier_threshold': 1.5,
        'dc_handle_categorical': True,
        'da_num_recommendations': 5,
        'da_iterations': 3,  # Run the aggregation generator 3 times
    }
    
    # Add custom model config if provided
    if llm_model:
        config_params['llm_config'] = llm_model
        
    config = PipelineConfig(**config_params)
    pipeline = DataAnalysisPipeline(dataset, config)
    return pipeline.run()