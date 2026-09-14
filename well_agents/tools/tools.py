from pathlib import Path
from agents import function_tool

from tools.output import WellTestContext, ZonalTestMemory, WellTestInterpretation, AnomalyInsights
from loguru import logger
from typing import List, Literal
from datetime import datetime
import csv
from pydantic import BaseModel

this_dir = Path(__file__).parent
file_path = this_dir.parent / "mem.txt"
MEMORY_FIELDS = [
    'Date',
    'WellName',
    'Anomaly',
    'AnomalyType',
    'WTLIQ',
    'WTOil',
    'WTTHP',
    'WTWCT',
    'Z1Status',
    'Z2Status',
    'Z3Status',
    'Z1BHP',
    'Z2BHP',
    'Z3BHP',
]

# Tools
@function_tool
def save_test_memory(welldata: WellTestContext, file_path: str, 
                    Anomaly: bool,
                    AnomalyType: str,
                    Z1Status: str,
                    Z2Status: str,
                    Z3Status: str) -> None:
    """
        Save the well test data record as a comma separated line in a text file.
        Args:
            Welldata: The well test data record to save.
            file_path: The path to the file where the data will be saved.
    """

    #file_path =  'agents/mem.txt'
    logger.info(f"Saving well test data in memory")
    fields = [
        welldata.Date,
        welldata.WellName,
        Anomaly,
        AnomalyType,
        welldata.WTLIQ,
        welldata.WTOil,
        welldata.WTTHP,
        welldata.WTWCT,
        Z1Status,
        Z2Status,
        Z3Status,
        welldata.Z1BHP,
        welldata.Z2BHP,
        welldata.Z3BHP        
    ]
    
    target_path = Path(file_path)
    should_write_header = not target_path.exists() or target_path.stat().st_size == 0

    with open(target_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if should_write_header:
            writer.writerow(MEMORY_FIELDS)
        writer.writerow(fields)
    
class WellTestData(BaseModel):
    Date: datetime
    WellName: str
    Anomaly: bool
    AnomalyType: str
    WTLIQ: float
    WTOil: float
    WTTHP: float
    WTWCT: float
    Z1Status: Literal["Open", "Closed"]
    Z2Status: Literal["Open", "Closed"]
    Z3Status: Literal["Open", "Closed"]
    Z1BHP: float
    Z2BHP: float
    Z3BHP: float


def load_well_memory_data(filepath: str) -> List[WellTestData]:
    well_data = []
    target_path = Path(filepath)

    if not target_path.exists() or target_path.stat().st_size == 0:
        return well_data
    
    with open(target_path, 'r', newline='') as file:
        csv_reader = csv.DictReader(file)
        
        for row in csv_reader:
            if not row or not row.get('Date'):
                continue

            data_dict = {
                'Date': datetime.strptime(row['Date'].strip(), '%Y-%m-%d'),
                'WellName': row['WellName'].strip(),
                'Anomaly': row['Anomaly'].strip().lower() == 'true',
                'AnomalyType': row['AnomalyType'].strip(),
                'WTLIQ': float(row['WTLIQ']),
                'WTOil': float(row['WTOil']),
                'WTTHP': float(row['WTTHP']),
                'WTWCT': float(row['WTWCT']),
                'Z1Status': row['Z1Status'].strip(),
                'Z2Status': row['Z2Status'].strip(),
                'Z3Status': row['Z3Status'].strip(),
                'Z1BHP': float(row['Z1BHP']),
                'Z2BHP': float(row['Z2BHP']),
                'Z3BHP': float(row['Z3BHP'])
            }
            well_data.append(WellTestData(**data_dict))
    
    return well_data