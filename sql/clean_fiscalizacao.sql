CREATE OR REPLACE TABLE fiscalizacao_limpa AS
SELECT
    COALESCE("Municipio", 'Não Informado') AS "Municipio",
    COALESCE("Sistema", 'Não Informado') AS "Sistema",
    "Subsistema",
    "ID Relatório",

    TRY_STRPTIME("Data Início Fiscalização", '%d/%m/%Y') AS "Data Início Fiscalização",
    TRY_STRPTIME("Data Fim Fiscalização", '%d/%m/%Y') AS "Data Fim Fiscalização",

    "Referência",
    "Número Item Fiscalizado",
    "Código Não Conformidade",
    "Item Não Conforme",


    TRY_STRPTIME("Data Notificação", '%d/%m/%Y') AS "Data Notificação",
    NULLIF("ID Notificação", '') AS "ID Notificação",

    TRY_STRPTIME("Data Advertencia", '%d/%m/%Y') AS "Data Advertencia",
    NULLIF("ID Advertencia", '') AS "ID Advertencia",

    TRY_STRPTIME("Data da Multa", '%d/%m/%Y') AS "Data da Multa",
    NULLIF("ID Multa", '') AS "ID Multa",

    TRY_STRPTIME("Data Limite Resolução", '%d/%m/%Y') AS "Data Limite Resolução",


    COALESCE("Condição", 'Status Desconhecido') AS "Condição",


    NULLIF("ID CAC", '') AS "ID CAC",
    NULLIF("Data Limite CAC", '') AS "Data Limite CAC"


FROM read_csv_auto('data/fiscalizacao.csv', delim=';', header=true, all_varchar=true);
