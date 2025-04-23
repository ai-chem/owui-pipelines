import enum
import json
from typing import Optional
from pydantic import (
    BaseModel,
    ValidationError,
    model_validator,
)


class PropertiesFilterMode(str, enum.Enum):
    gt = 'gt'
    lt = 'lt'
    both = 'both'


class PropertiesFilter(BaseModel):
    property_name: str
    less_than: Optional[float | int] = None
    greater_than: Optional[float | int] = None
    mode: PropertiesFilterMode

    @model_validator(mode='before')
    def set_filter_mode(self):
        set_keys = self.keys()
        if 'less_than' in set_keys and 'greater_than' in set_keys:
            self['mode'] = PropertiesFilterMode.both
        elif 'less_than' in set_keys:
            self['mode'] = PropertiesFilterMode.lt
        elif 'greater_than' in set_keys:
            self['mode'] = PropertiesFilterMode.gt
        else:
            raise ValidationError('At least one filter bound must be set')
        return self


class QueryCategory(str, enum.Enum):
    general = "general"
    synthesis = "synthesis"
    properties = "properties"


class CategorizationResult(BaseModel):
    category: QueryCategory
    content: str
    __transformed_content: Optional[list[dict]] = None

    @model_validator(mode="after")
    def parse_content(self):
        if self.category == QueryCategory.properties:
            self.__transformed_content = json.loads(self.content)
        return self

    def get_transformed_content(self):
        return self.__transformed_content
