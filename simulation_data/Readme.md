## Run
0. Use environment `oecsim`
    ```
    conda activate oecsim
    ```
1. Put all the access related txt files "Place-<Place_name>-To-Satellite-All_xxxx.txt" into a folder named `access_records`.
2. Put all the solar intensity related txt files "Sat_<satellite_name>_solar.txt" into a folder `solar`.
3. Run `parse_access_reports.py` with
    ```
    python parse_access_reports.py "results/access_records" --save-json "parsed_access.json"
    ```
4. Run `parse_solar_reports.py` with
    ```
    python parse_solar_reports.py "results/solar" --save-json "solar_parsed.json"
    ```
5. Run `build_testcase.py` with
    ```
    python build_testcase.py
    ```
    The output will be a json file named `test_case.json`
