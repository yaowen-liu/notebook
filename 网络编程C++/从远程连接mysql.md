在执行`GRANT`语句之前，需要先创建`liuyaowen`用户。

你可以使用以下命令来创建一个名为`liuyaowen`的用户：

```sql
CREATE USER 'liuyaowen'@'%' IDENTIFIED BY 'password';
```

将`password`替换为你想要设置的实际密码。

然后，你可以使用`GRANT`语句给该用户授予所有权限：

```sql
GRANT ALL PRIVILEGES ON *.* TO 'liuyaowen'@'%' WITH GRANT OPTION;
```

这样，`liuyaowen`用户就被创建并被授予了所有数据库的所有权限。

![image-20240107210028802](https://raw.githubusercontent.com/yaowen-liu/notebook/master/data/202401072100879.png)

用户名：liuyaowen，密码：88888888