import logging
import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field,constr, confloat,conlist
import pandas as pd
from typing import List,Optional
import joblib
import lightgbm
import sklearn
from enum import Enum




app = FastAPI()
model = joblib.load('lgb_model_main.joblib')


categorical_features = [
    'NAME_CONTRACT_TYPE',
    'CODE_GENDER',
    'NAME_INCOME_TYPE',
    'NAME_EDUCATION_TYPE',
    'NAME_FAMILY_STATUS',
    'OCCUPATION_TYPE',
    'ORGANIZATION_TYPE',
]

class ContractType(str, Enum):
    Cash_loans = "Cash loans"
    Revolving_loans = "Revolving loans"

class Gender(str, Enum):
    Male = "M"
    Female = "F"
    XNA ="XNA"

class IncomeType(str, Enum):
    Working = "Working"
    Other = "Other"
    Commercial_associate = "Commercial associate"
    Pensioner = "Pensioner"

class EducationType(str, Enum):
    Other = "Other"
    Higher_education = "Higher education"
    Secondary = "Secondary / secondary special"

class FamilyStatus(str, Enum):
    Civil_marriage = "Civil marriage"
    Married = "Married"
    Single = "Single / not married"
    Other = "Other"

class OccupationType(str, Enum):
    Laborers = "Laborers"
    Sales_staff = "Sales staff"
    Core_staff = "Core staff"
    Managers = "Managers"
    Drivers = "Drivers"
    Other = "Other"

class OrganizationType(str, Enum):
    Business_Entity = "Business Entity Type 3"
    Other = "Other"
    XNA = "XNA"
    Self_employed = "Self-employed"










class PredictionInput(BaseModel):
    AMT_INCOME_TOTAL: confloat(ge=0)
    AMT_CREDIT: confloat(ge=0)
    REGION_POPULATION_RELATIVE: confloat(ge=0)
    DAYS_REGISTRATION: int
    DAYS_BIRTH: int
    DAYS_ID_PUBLISH: int
    FLAG_WORK_PHONE: int
    FLAG_PHONE: int
    REGION_RATING_CLIENT_W_CITY: int
    REG_CITY_NOT_WORK_CITY: int
    FLAG_DOCUMENT_3: int
    NAME_CONTRACT_TYPE: ContractType
    CODE_GENDER: Gender
    FLAG_OWN_CAR: int
    NAME_INCOME_TYPE: IncomeType
    NAME_EDUCATION_TYPE: EducationType
    NAME_FAMILY_STATUS: FamilyStatus
    OCCUPATION_TYPE: OccupationType
    ORGANIZATION_TYPE: OrganizationType
    CREDIT_ACTIVE_Active_count_Bureau: Optional[int] = None
    CREDIT_ACTIVE_Closed_count_Bureau: Optional[int] = None
    DAYS_CREDIT_Bureau: Optional[int] = None
    AMT_INSTALMENT_mean_HCredit_installments: Optional[int] = None
    DAYS_INSTALMENT_mean_HCredit_installments: Optional[int] = None
    NUM_INSTALMENT_NUMBER_mean_HCredit_installments: Optional[int] = None
    NUM_INSTALMENT_VERSION_mean_HCredit_installments: Optional[int] = None
    NAME_CONTRACT_STATUS_Active_count_pos_cash: Optional[int] = None
    NAME_CONTRACT_STATUS_Completed_count_pos_cash: Optional[int] = None
    SK_DPD_DEF_pos_cash: Optional[int] = None
    NAME_CONTRACT_STATUS_Refused_count_HCredit_PApp: Optional[int] = None
    NAME_GOODS_CATEGORY_Other_count_HCredit_PApp: Optional[int] = None
    NAME_PORTFOLIO_Cash_count_HCredit_PApp: Optional[int] = None
    NAME_PRODUCT_TYPE_walk_in_count_HCredit_PApp: Optional[int] = None
    NAME_SELLER_INDUSTRY_Other_count_HCredit_PApp: Optional[int] = None
    NAME_YIELD_GROUP_high_count_HCredit_PApp: Optional[int] = None
    NAME_YIELD_GROUP_low_action_count_HCredit_PApp: Optional[int] = None
    AMT_CREDIT_HCredit_PApp: Optional[int] = None
    SELLERPLACE_AREA_HCredit_PApp: Optional[int] = None

    class Config:
        @classmethod
        def validate_values(cls, values):
            for field_name, field_value in values.items():
                if isinstance(getattr(cls, field_name, None), Enum):
                    enum_values = [item.value for item in getattr(cls, field_name)]
                    if field_name == 'CODE_GENDER' and field_value not in enum_values:
                        setattr(cls, field_name, cls._str_to_enum(field_name, "XNA"))
                    elif field_value not in enum_values:
                        setattr(cls, field_name, cls._str_to_enum(field_name, "Other"))

        @staticmethod
        def _str_to_enum(field_name, default):
            return getattr(globals()[field_name], default)

    @classmethod
    def __get_validators__(cls):
        yield cls.validate_values







@app.post("/predict")
def predict(data: PredictionInput):
    try:
        # Convert Pydantic model to a dictionary
        input_dict = data.dict()

        # Convert dictionary to a pandas DataFrame
        input_df = pd.DataFrame([input_dict])

        # Convert categorical features to 'category' type
        for feature in categorical_features:
            input_df[feature] = input_df[feature].astype('category')

        # Make predictions using the loaded model
        predictions = model.predict_proba(input_df, categorical_feature=categorical_features)[:, 1]

        # Placeholder response for demonstration
        response = {"Probability for this credit to be defaulted is: ": predictions[0]}  # Extract the probability for class 1

        return response
    except Exception as e:
        # Handle exceptions and return an HTTP 500 error
        raise HTTPException(status_code=500, detail=str(e))