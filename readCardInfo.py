# This code read in card information from prepared excel sheet
# information includes: card name, type, rarity, cost
# how to use:
# import readCardInfo
# CardInfo = readCardInfo.cardObj(xlsxName) 

from xlrd import open_workbook
import xlrd
class cardObj(object):
    def __init__(self, ID, NAME, TYPE, RARITY, COST):
        self.ID = ID
        self.NAME = NAME
        self.TYPE = TYPE
        self.RARITY = RARITY
        self.COST = COST

    def __str__(self):
        return("Card object:\n"
               "  ID = {0}\n"
               "  NAME = {1}\n"
               "  TYPE = {2}\n"
               "  RARITY = {3}\n"
               "  COST = {4} \n"
               .format(int(float(self.ID)), self.NAME, self.TYPE,
                        self.RARITY, int(float(self.COST))))

def readCARD(filename):
  wb = open_workbook(filename)
  items = {}
  for sheet in wb.sheets():
      number_of_rows = sheet.nrows
      number_of_columns = sheet.ncols
      rows = []
      for row in range(1, number_of_rows):
          values = []
          for col in range(number_of_columns):
              value  = str(sheet.cell(row,col).value)
              try:
                  value = str(xlrd.xldate_as_tuple(value, 0))
              except ValueError:
                  pass
              finally:
                  values.append(value)
          item = cardObj(*values)
          items[item.NAME] = item
  return items
