#!/usr/bin/env python3
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional


class FunctionParameter(BaseModel):
    type: str = Field(..., description="parameters type")
    description: Optional[str] = Field(
        None, description="Description")
    enum: Optional[List[str]] = Field(None, description="Valeur Possibels")


class FunctionDefinition(BaseModel):
    name: str = Field(..., description="function name")
    description: str = Field(..., description="description")
    parameters: Dict[str, FunctionParameter] = Field(default_factory=dict)
    required: List[str] = Field(default_factory=list)
    returns: Optional[Dict[str, Any]] = Field(None)


class FunctionCall(BaseModel):
    prompt: str = Field(..., description="Original question")
    name: str = Field(..., description="title or name of function")
    parameters: Dict[str, Any] = Field(default_factory=dict)
    """ default factory signifie que pydantic
        cree une dict ou liste independant
        chaque fois pour chaque appel
    """
