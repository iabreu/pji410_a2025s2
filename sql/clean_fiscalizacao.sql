CREATE OR REPLACE TABLE fiscalizacao_limpa AS
SELECT
    COALESCE("Municipio", 'Não Informado') AS "Municipio",
    COALESCE("Sistema", 'Não Informado') AS "Sistema",
    "Subsistema",
    "ID Relatório",

    COALESCE(TRY_STRPTIME("Data Início Fiscalização", '%d/%m/%Y'), TRY_STRPTIME("Data Início Fiscalização", '%m/%d/%Y')) AS "Data Início Fiscalização",
    COALESCE(TRY_STRPTIME("Data Fim Fiscalização", '%d/%m/%Y'), TRY_STRPTIME("Data Fim Fiscalização", '%m/%d/%Y')) AS "Data Fim Fiscalização",

    "Referência",
    "Número Item Fiscalizado",
    "Código Não Conformidade",
    "Item Não Conforme",

    COALESCE(TRY_STRPTIME("Data Notificação", '%d/%m/%Y'), TRY_STRPTIME("Data Notificação", '%m/%d/%Y')) AS "Data Notificação",
    NULLIF("ID Notificação", '') AS "ID Notificação",

    COALESCE(TRY_STRPTIME("Data Advertencia", '%d/%m/%Y'), TRY_STRPTIME("Data Advertencia", '%m/%d/%Y')) AS "Data Advertencia",
    NULLIF("ID Advertencia", '') AS "ID Advertencia",

    COALESCE(TRY_STRPTIME("Data da Multa", '%d/%m/%Y'), TRY_STRPTIME("Data da Multa", '%m/%d/%Y')) AS "Data da Multa",
    NULLIF("ID Multa", '') AS "ID Multa",

    COALESCE(TRY_STRPTIME("Data Limite Resolução", '%d/%m/%Y'), TRY_STRPTIME("Data Limite Resolução", '%m/%d/%Y')) AS "Data Limite Resolução",

    LOWER(TRIM(COALESCE("Condição", 'desconhecido'))) AS "Condição",

    NULLIF("ID CAC", '') AS "ID CAC",

    COALESCE(TRY_STRPTIME("Data Limite CAC", '%d/%m/%Y'), TRY_STRPTIME("Data Limite CAC", '%m/%d/%Y')) AS "Data Limite CAC"

FROM read_csv_auto('{{CSV_PATH}}', delim=';', header=true, all_varchar=true);