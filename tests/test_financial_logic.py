from generation.rag import query_profile,query_target_metadata,extract_financial_answer

queries=[
    "What financial results are reported for FY 2024-25?",
    "What happened to the companys total income between FY 2023-24 and FY 2024-25?",
    "What was the total income for FY 2024-25?",
]

financial_table=(
    "The Standalone and Consolidated financial results for the year ended March 31, 2025 "
    "are given below: (` In lakhs) Particulars STANDALONE CONSOLIDATED FY 2024-25 FY 2023-24 "
    "FY 2024-25 FY 2023-24 Total Income 31,455.43 24,796.96 37,569.16 31,813.11 "
    "Total Expenditure 27,897.66 26,420.79 57,566.17 50,443.56"
)

results=[{
    "document":financial_table,
    "metadata":{
        "page_number":5,
        "section":"1. FINANCIAL RESULTS",
        "document_id":"tata_annual_report_2024_25",
    },
}]

for query in queries:
    print("\n"+"="*80)
    print("QUERY:",query)
    print("PROFILE:",query_profile(query))
    print("TARGET:",query_target_metadata(query))
    print("ANSWER:",extract_financial_answer(query,results))