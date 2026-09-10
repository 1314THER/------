#!/usr/bin/env node
/**
 * 生成 result2.xlsx / result3.xlsx / result4.xlsx。
 *
 * 输入: 输出/result2.bin、result3.bin、result4.bin、result_meta.json
 *       （由 代码/prepare_outputs.py 生成）
 * 输出: 输出/result2.xlsx、result3.xlsx、result4.xlsx
 *
 * 格式遵循题面附件 3 的模板:
 *   - A 列为时间 (单位: s)，第 1 行为到药材中心的距离 (单位: cm)
 *   - result2 含 "温度" 与 "水分浓度" 两个工作表
 *   - result3、result4 只含水分浓度；result4 末列为 "药材表面"
 *   - 超出当前半径的位置留空（问题四收缩后 R < 2 cm）
 *   - 所有数值保留四位小数
 *
 * 运行: node 代码/build_results234.mjs [2|3|4]
 */

import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const PROJECT = path.resolve(HERE, "..");
const OUT = path.join(PROJECT, "输出");

const FONT = "Arial";
const HEADER_LABEL = "时间\\到药材中心的距离";
const NUMFMT = "0.0000";

async function readF64(file) {
  const buf = await fs.readFile(file);
  const ab = new ArrayBuffer(buf.byteLength);
  Buffer.from(ab).set(buf);
  return new Float64Array(ab);
}

/** 写入一个工作表：A 列时间，其后为各径向位置，可选再追加一列（药材表面）。 */
function fillSheet(workbook, name, times, grid, extraName, extra) {
  const sheet = workbook.worksheets.add(name);
  const headers = [HEADER_LABEL, ...positions];
  if (extraName) headers.push(extraName);
  const nCols = headers.length;
  const nRows = times.length;
  const round4 = (v) =>
    Number.isFinite(v) ? Math.round(v * 1e4) / 1e4 : null;

  const values = new Array(nRows + 1);
  values[0] = headers;
  for (let i = 0; i < nRows; i++) {
    const row = new Array(nCols);
    row[0] = times[i];
    const base = i * nPos;
    for (let j = 0; j < nPos; j++) {
      row[1 + j] = round4(grid[base + j]);
    }
    if (extraName) row[1 + nPos] = round4(extra[i]);
    values[i + 1] = row;
  }

  const body = sheet.getRangeByIndexes(0, 0, nRows + 1, nCols);
  body.values = values;
  body.format.numberFormat = NUMFMT;
  body.format.font = { name: FONT, size: 10 };

  const header = sheet.getRangeByIndexes(0, 0, 1, nCols);
  header.format.fill = "#D9D9D9";
  header.format.font = { name: FONT, bold: true, size: 10 };
  header.format.borders = { preset: "outside", style: "thin", color: "#808080" };

  sheet.getRange("A:A").format.columnWidth = 23;
  sheet.getRange("A1").format.alignment = "left";
  sheet.getRangeByIndexes(0, 1, 1, nCols - 1).format.columnWidth = 10;
  sheet.freezePanes.freezeRows(1);
  sheet.freezePanes.freezeColumns(1);
  sheet.showGridLines = false;
  return sheet;
}

async function save(workbook, name) {
  workbook.recalculate();
  const blob = await SpreadsheetFile.exportXlsx(workbook);
  const target = path.join(OUT, name);
  await blob.save(target);
  console.log(`已生成: ${target}`);
}

const meta = JSON.parse(
  await fs.readFile(path.join(OUT, "result_meta.json"), "utf8"),
);
const positions = meta.positions_cm;
const which = process.argv[2];
const nPos = positions.length;

if (!which || which === "2") {
  const arr = await readF64(path.join(OUT, "result2.bin"));
  const n = meta.n2;
  const times = arr.subarray(0, n);
  const temp = arr.subarray(n, n + n * nPos);
  const conc = arr.subarray(n + n * nPos, n + 2 * n * nPos);
  const wb = Workbook.create();
  fillSheet(wb, "温度", times, temp);
  fillSheet(wb, "水分浓度", times, conc);
  await save(wb, "result2.xlsx");
}

if (!which || which === "3") {
  const arr = await readF64(path.join(OUT, "result3.bin"));
  const n = meta.n3;
  const times = arr.subarray(0, n);
  const conc = arr.subarray(n, n + n * nPos);
  const wb = Workbook.create();
  fillSheet(wb, "Sheet1", times, conc);
  await save(wb, "result3.xlsx");
}

if (!which || which === "4") {
  const arr = await readF64(path.join(OUT, "result4.bin"));
  const n = meta.n4;
  const times = arr.subarray(0, n);
  const conc = arr.subarray(n, n + n * nPos);
  const surf = arr.subarray(n + n * nPos, n + n * nPos + n);
  const wb = Workbook.create();
  fillSheet(wb, "Sheet1", times, conc, "药材表面", surf);
  await save(wb, "result4.xlsx");
}
