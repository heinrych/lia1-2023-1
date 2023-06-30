CREATE TABLE cliente 
( 
 cli_id SERIAL PRIMARY KEY,  
 cli_nome VARCHAR(80),
 cli_endereco VARCHAR(120)
);

CREATE TABLE livro 
( 
 liv_id SERIAL PRIMARY KEY,  
 liv_titulo VARCHAR(100),  
 liv_isbn VARCHAR(11)
);

CREATE TABLE venda 
( 
 ven_id SERIAL PRIMARY KEY,  
 ven_frete FLOAT,  
 ven_valor FLOAT,
 liv_id INT,
 cli_id INT
); 

ALTER TABLE venda ADD FOREIGN KEY(liv_id) REFERENCES livro (liv_id);
ALTER TABLE venda ADD FOREIGN KEY(cli_id) REFERENCES cliente (cli_id)

INSERT INTO livro (liv_titulo, liv_isbn) VALUES ('d', '654')
SELECT * FROM livro

INSERT INTO cliente (cli_nome, cli_endereco) VALUES ('Maria', 'Rua 2')
SELECT * FROM cliente

INSERT INTO venda (ven_frete, ven_valor, liv_id, cli_id) VALUES (12.0, 22.0, 1, 2)
SELECT * FROM venda

SELECT * FROM venda INNER JOIN cliente ON cliente.cli_nome = 'Maria'
