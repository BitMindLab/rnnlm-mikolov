QT = core
SOURCES = main.cpp qrnnlm.cpp 
HEADERS = qrnnlm.h 
QMAKE_CXXFLAGS += -msse2 
TEMPLATE = app
CONFIG += release
