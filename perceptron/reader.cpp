#include <sstream>
#include <fstream>
#include <cstdlib>
#include <cctype>
#include "reader.h"

arma::mat Reader::readCSV(std::string name)
{
  std::ifstream in(name.c_str());
  if (! in.is_open())
  {
    std::cerr << "error opening the file: " << name << "\n";
    exit(1);
  }
  
  arma::mat A;
  std::string temp,token;
  std::stringstream ss;
  int rowCount = 0;  
  while( std::getline(in, temp) ){
    ss.clear();
    ss << temp;
    while(std::getline(ss, token, ','))
    {
       A << stod(token);
    }
     
     A << arma::endr;
  }//grab a line in csv files
  A.print();
  return A;
}
