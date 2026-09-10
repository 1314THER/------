#!/usr/bin/env node
/**
 * 由问题一的求解结果生成 result1.xlsx。
 *
 * 输入: 输出/result1_data.json   （由 solve_p1.py --save 自动导出）
 * 输出: 输出/result1.xlsx
 *
 * 工作表格式遵循题面附件 3 的模板:
 *   - "温度" 与 "水分浓度" 两个工作表
 *   - A 列为时间 (单位: s)，第 1 行为到药材中心的距离 (单位: cm)
 *   - 所有数值保留四位小数
 *
 * 运行:
 *   PY=/Users/sqz/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3
 *   $PY 代码/solve_p1.py --save 输出/result1_data.npz
 *   node 代码/build_result1.mjs
 */

import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const PROJECT = path.resolve(HERE, "..");
const INPUT = path.join(PROJECT, "输出", "result1_data.json");
const OUTPUT = path.join(PROJECT, "输出", "result1.xlsx");

const FONT = "Arial";
const HEADER_LABEL = "时间\\到药材中心的距离";
const NUMBER_FORMAT = "0.0000";

const data = JSON.parse(await fs.readFile(INPUT, "utf8"));
const radiusCm = data.radius_cm;
const times = data.times;
const nCols = radiusCm.length + 1;

// 每一行必须严格对应一个时刻，避免错行
for (const key of ["temperature", "moisture"]) {
  if (data[key].length !== times.length) {
    throw new Error(
      `${key} 行数 ${data[key].length} 与时间点数 ${times.length} 不一致`,
    );
  }
}

function sheetValues(rows) {
  return [
    [HEADER_LABEL, ...radiusCm],
    ...times.map((t, i) => [t, ...rows[i]]),
  ];
}

function addSheet(workbook, name, rows) {
  const sheet = workbook.worksheets.add(name);
  const values = sheetValues(rows);
  const nRows = values.length;

  const body = sheet.getRangeByIndexes(0, 0, nRows, nCols);
  body.values = values;
  body.format.numberFormat = NUMBER_FORMAT;

  const header = sheet.getRangeByIndexes(0, 0, 1, nCols);
  header.format.fill = "#D9D9D9";
  header.format.font = { name: FONT, bold: true, size: 10 };
  header.format.borders = { preset: "outside", style: "thin", color: "#808080" };

  sheet.getRangeByIndexes(1, 1, nRows - 1, nCols - 1).format.font = {
    name: FONT,
    size: 10,
  };
  sheet.getRangeByIndexes(1, 0, nRows - 1, 1).format.font = {
    name: FONT,
    size: 10,
  };

  // A 列需容纳 "时间\到药材中心的距离" 表头，否则会被右侧单元格挤掉
  sheet.getRange("A:A").format.columnWidth = 23;
  sheet.getRange("A1").format.alignment = "left";
  sheet.getRangeByIndexes(0, 1, 1, nCols - 1).format.columnWidth = 10;

  sheet.freezePanes.freezeRows(1);
  sheet.freezePanes.freezeColumns(1);
  sheet.showGridLines = false;
  return sheet;
}

const workbook = Workbook.create();
addSheet(workbook, "温度", data.temperature);
addSheet(workbook, "水分浓度", data.moisture);

workbook.recalculate();

await fs.mkdir(path.dirname(OUTPUT), { recursive: true });
const blob = await SpreadsheetFile.exportXlsx(workbook);
await blob.save(OUTPUT);

const check = await workbook.inspect({
  kind: "region",
  sheetId: "温度",
  range: "A1:E5",
  maxChars: 1200,
});
console.log(check.ndjson);
console.log(`已生成: ${OUTPUT}`);
