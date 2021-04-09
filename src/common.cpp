#include "common.h"
#include <string>
#include <vector>
#include <iostream>


std::string prefix = "";
std::vector<std::string> v;

void print_prefix(std::string s)
{
  prefix = s;
}

void print_enter(std::string s)
{
  cout << "\033[1;32mentering:" << s << "\033[0m" << endl;
  v.push_back(s);
}

void print_leave()
{
  cout << "\033[1;31mleaving:" << v.back() << "\033[0m" << endl;
  v.pop_back();
}

void _print_prefix()
{
  cout << "\033[1;33m" << prefix;
  for (std::string &i : v)
    cout << i;
}

void _print_suffix()
{
  cout << "\033[0m" << endl;
}

void print(std::string s)
{
  _print_prefix();
  cout << s;
  _print_suffix();
}

void print(int a)
{
  _print_prefix();
  cout << a;
  _print_suffix();
}
