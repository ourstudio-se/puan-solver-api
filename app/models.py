# models.py
import numpy as np
from typing import List, Dict, Optional
from pydantic import BaseModel, Field, model_validator
from enum import Enum

class Solver(str, Enum):
    default = "default"

class Direction(str, Enum):
    maximize = "maximize"
    minimize = "minimize"

class Shape(BaseModel):

    nrows: int = Field(..., description="Number of rows")
    ncols: int = Field(..., description="Number of columns")

    def __hash__(self): 
        return hash((self.nrows, self.ncols))
    
    def as_tuple(self) -> tuple:
        return (self.nrows, self.ncols)

class SparseMatrix(BaseModel):
    def __hash__(self): 
        return hash((
            type(self), 
            tuple(self.rows), 
            tuple(self.cols), 
            tuple(self.vals), 
            self.shape,
        ))
    
    rows: List[int] = Field(..., description="Row indices of the sparse matrix")
    cols: List[int] = Field(..., description="Column indices of the sparse matrix")
    vals: List[int] = Field(..., description="Values of the sparse matrix")
    shape: Shape = Field(..., description="Shape of the sparse matrix")

    def to_numpy(self) -> np.ndarray:

        # Depend dtype on the maximum value in the matrix
        mx = max(map(abs, self.vals), default=1) * 2
        dtype = np.int8 if mx < 256 else (np.int16 if mx < 32767 else np.int32)
    
        # Create an empty matrix filled with zeros
        dense_matrix = np.zeros(self.shape.as_tuple(), dtype=dtype)
        
        # Assign values to the corresponding positions
        dense_matrix[self.rows, self.cols] = self.vals
        
        return dense_matrix
    
    @classmethod
    def from_numpy(cls, matrix: np.ndarray) -> 'SparseMatrix':
        rows, cols = np.nonzero(matrix)
        vals = matrix[rows, cols]
        return cls(rows=rows.tolist(), cols=cols.tolist(), vals=vals.tolist(), shape=dict(zip(("nrows", "ncols"), matrix.shape)))


class Variable(BaseModel):
    id: str = Field(..., description="Unique identifier of the variable")
    bound: List[int] = Field([0, 1], description="Bounds of the variable, e.g., [0, 1] for binary variables")

class Polyhedron(BaseModel):
    def __hash__(self):  # make hashable BaseModel subclass
        return hash((
            type(self), 
            self.A, 
            tuple(self.b),
        ))
    
    A: SparseMatrix = Field(..., description="`A` sparse matrix of Polyhedron")
    b: List[int] = Field(..., description="`b` vector of Polyhedron")
    variables: List[Variable] = Field([], description="List of variables in the Polyhedron") 

class Model(BaseModel):
    def __hash__(self):  # make hashable BaseModel subclass
        return hash((
            type(self), 
            self.id, 
            self.polyhedron, 
            tuple(self.columns), 
            tuple(self.rows), 
            tuple(self.intvars),
        ))
    
    id: Optional[str] = Field(None, description="Unique identifier of Polyhedron")
    polyhedron: Polyhedron = Field(..., description="Polyhedron")
    columns: List[str] = Field([], description="Column names of Polyhedron")
    rows: List[str] = Field([], description="Row names of Polyhedron")
    intvars: List[str] = Field([], description="Integer variables of Polyhedron (boolean is assumed otherwise)")

    def tobytes(self) -> bytes:
        return self.model_dump_json().encode('utf-8')
    
    @model_validator(mode='before')
    def set_columns_from_variables(data: dict):
        poly_vars = data.get('polyhedron', {}).get('variables', [])
        poly_A_shape = data.get('polyhedron', {}).get('A', {}).get('shape', {})
        columns = data.get('columns', [])
        rows = data.get('rows', [])
        intvars = data.get('intvars', [])

        if not columns and poly_vars:
            data['columns'] = [var.get('id') for var in poly_vars]
        
        if not rows and poly_A_shape.get('nrows', 0) > 0:
            data['rows'] = [f"row_{i}" for i in range(poly_A_shape.get('nrows', 0))]
        
        if not intvars:
            data['intvars'] = [var.get('id') for var in poly_vars if not (var.get('bound', [0, 1])[0] == 0 and var.get('bound', [0, 1])[1] == 1)]
        
        return data

    @model_validator(mode='after')
    def check_columns(self):
        if len(set(self.columns)) != len(self.columns):
            raise ValueError("Columns must be unique")
        
        if len(self.columns) != self.polyhedron.A.shape.ncols:
            raise ValueError("Number of columns should match the number of columns in `A`")
        
        return self

class CreateModelResponse(BaseModel):
    key: str = Field(..., description="Unique identifier for the created polyhedron")

class RetrieveModelResponse(BaseModel):
    model: Model = Field(..., description="Polyhedron model")

class SolveILPRequest(BaseModel):
    direction: Direction = Field(..., description="Direction of the optimization")
    objectives: List[Dict[str, float]] = Field(..., description="Objectives to maximize or minimize")
    solver: Optional[Solver] = Field(Solver.default, description="Solver to use")

class ILPSolution(BaseModel):
    solution: Optional[Dict[str, int]] = Field(..., description="Values of decision variables that optimize the objectives")
    error: Optional[str] = Field(None, description="Error details")

class SolveILPResponse(BaseModel):
    solutions: List[ILPSolution] = Field(..., description="List of ILP solutions")

class ErrorResponse(BaseModel):
    error: str = Field(..., description="Error details")

class Problem(BaseModel):
    model: Model = Field(..., description="Polyhedron model")
    objectives: List[Dict[str, float]] = Field(..., description="Objectives to maximize or minimize")
    direction: Direction = Field(..., description="Direction of the optimization")
    solver: Solver = Field(Solver.default, description="Solver to use")

class ReduceResponse(BaseModel):
    reduced_model: Model = Field(..., description="Reduced polyhedron model")
    consequence: Dict[str, int] = Field(..., description="Fixed variable consequence of the reduction")
