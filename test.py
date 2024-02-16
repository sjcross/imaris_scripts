import ImarisLib

# Get Imaris instance
imaris_lib = ImarisLib.ImarisLib()
imaris_application = imaris_lib.GetApplication(1)
if imaris_application is None:
    print('Imaris can not be found')
    quit

sel = imaris_application.GetSurpassSelection()
raw_input("tofggdfg")
print(sel.GetName())