import enum
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
    less_than: Optional[float | int]
    greater_than: Optional[float | int]
    mode: PropertiesFilterMode

    @model_validator(mode='before')
    def set_filter_mode(self):
        print(self)
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
    content: str | list[dict[str, str | float | int | None]]
