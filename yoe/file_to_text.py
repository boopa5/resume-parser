# import PyPDF2
# import docx2txt
# import os
#
# import resume_ner as yoe
#
#
# def get_file_text(filename):
#     try:
#         if filename[-1] == 'f':
#             pdf_file_obj = open(filename, 'rb')
#             pdf_reader = PyPDF2.PdfFileReader(pdf_file_obj)
#
#             res = ''
#             for i in range(pdf_reader.getNumPages()):
#                 res += pdf_reader.getPage(0).extractText()
#
#             return res
#
#         elif filename[-1] == 'x':
#             text = docx2txt.process(filename)
#             return text
#
#         else:
#             return ''
#     except:
#         print('failed')
#
#
# path = "./Resumes"
#
# file_names = []
# for root, d_names, f_names in os.walk(path):
#     for f in f_names:
#         file_names.append(os.path.join(root, f))
#
# actual = [2.83, 15.08, 3.42, 4.83]
#
# data_text_list = [get_file_text(filename) for filename in file_names if filename[-1] != 'c']
#
# res = [(yoe.predict_yoe(data)) for data in data_text_list]
# error = []
# for i in range(len(actual)):
#     error.append({
#         'percent_error': (float(res[i]['years']) - actual[i]) / actual[i] * 100,
#         'expected': actual[i],
#         'got': float(res[i]['years'])
#     })
