## 参考链接：

https://blog.csdn.net/Yubing792289314/article/details/104820850

## Windows Clion工程连接Mysql数据库

​		windows系统上，C++项目要连接Mysql，有以下几种方法：（不管是本地的还是外部服务器，差别也就只是填写的ip地址）

​		官网提供的三种连接的方式：

​		1.以纯C的方式，使用Mysql Server 8.0里的libmysql.lib和libmysql.dll动态库

​		2.以C++11的方式，使用Connector/C++ 8.0，支持以JDBC的方式连接数据库

​		3.以ODBC的方式（目前我只会用 Qt 以 ODBC 的方式连接SqlServer）

​		纯C的好处是移植性比C++11的要好，且更快；C++11的好处则是面向对象，写起来舒服，写过Java的不愁写C++，但是前提是必须提前安装过OpenSSL库

## 1.以纯c的方式连接Mysql

1.找到mysql安装路径下的这两个文件，复制他们的路径到自己的工程的CMakeList.txt里

![image-20240107195309894](https://raw.githubusercontent.com/yaowen-liu/notebook/master/data/202401071953051.png)

2.将下面的四行代码加入到自己的工程的CMakeList.txt里

```cmake
include_directories(D:\\mysql_8.1.0\\mysql-8.1.0-winx64\\include)//包含include文件夹
link_directories(D:\\mysql_8.1.0\\mysql-8.1.0-winx64\\lib)//包含lib文件夹
link_libraries(libmysql)


target_link_libraries(${PROJECT_NAME} libmysql)//链接
```

![image-20240107200710219](https://raw.githubusercontent.com/yaowen-liu/notebook/master/data/202401072007321.png)