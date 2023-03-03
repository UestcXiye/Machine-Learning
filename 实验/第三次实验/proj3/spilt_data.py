import xlrd as xd

file_train_path = 'data/abalone_train.xlsx'
file_train_xlsx = xd.open_workbook(file_train_path)
file_train_sheet = file_train_xlsx.sheet_by_name('Sheet1')
x_train = []
y_train = []
for row in range(file_train_sheet.nrows):
    x_data = []
    for col in range(file_train_sheet.ncols):
        if col == 0:
            if file_train_sheet.cell_value(row, col) == 'M':
                y_train.append(1)
            elif file_train_sheet.cell_value(row, col) == 'F':
                y_train.append(-1)
            else:
                y_train.append(0)
        else:
            x_data.append(file_train_sheet.cell_value(row, col))

    x_train.append(list(x_data))

file_test_path = 'data/abalone_test.xlsx'
file_test_xlsx = xd.open_workbook(file_test_path)
file_test_sheet = file_test_xlsx.sheet_by_name('Sheet1')
x_test = []
y_test = []
for row in range(file_test_sheet.nrows):
    x_data = []
    for col in range(file_test_sheet.ncols):
        if col == 0:
            if file_test_sheet.cell_value(row, col) == 'M':
                y_test.append(1)
            elif file_test_sheet.cell_value(row, col) == 'F':
                y_test.append(-1)
            else:
                y_test.append(0)
        else:
            x_data.append(file_test_sheet.cell_value(row, col))

    x_test.append(list(x_data))

# print(x_train)
# print(y_train)
# print(x_test)
# print(y_test)
