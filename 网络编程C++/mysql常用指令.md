```c++

/**********建库********/
create database 数据库名; // 建库
use 数据库名; // 选择数据库
create table 数据表名 (column_name column_type); // 建表
insert into 数据表名 ( field1, field2,...fieldN ) values ( value1, value2,...valueN ); // 插入数据
select column_name,column_name from table_name [where Clause] [LIMIT N][ OFFSET M]; // 查询数据
delete from 表名 where 条件; //删除数据
update 数据表名 set 字段名 = 值 where 条件; // 修改数据
alter table 数据表名 change 原字段名 新字段名 新字段的类型 ; //修改字段

```

