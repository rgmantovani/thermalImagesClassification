# Deitado
for i in {1..280}
do
  echo "Deitado "${i}".csv" 
  mv "Deitado "${i}".csv" "Deitado_"${i}".csv"
done

# Em pé
for i in {1..280}
do
  echo "Em pé "${i}".csv" 
  mv "Em pé "${i}".csv" "Em_pe_"${i}".csv"
done

