# Ahora definimos una funcion para hacer la carga de las bases de datos
# La funcion agregar recibe una ubicacion de archivos, y la extension
def agregar(folder_path='archivos', extension=".xlsx"): 
   
    #Definimos un data frame vacio para ir almacenando cada excel
    all_data = pd.DataFrame()
   
    #   Ahora con un loop se va a recorrer cada archivo que este en la carpeta archivo
    for f in glob.glob(os.getcwd()+'/'+folder_path+'/*'+extension):
    
        dfs = pd.read_csv(f, header=0)
        dfs = dfs.drop(columns='Unnamed: 0')
               
        all_data = all_data.append(dfs, ignore_index=True)
        
        
    all_data.columns = ['artist', 'genre', 'title', 'lyrics']
    return all_data