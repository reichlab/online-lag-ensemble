// Common utilities for js scripts

const fs = require('fs-extra')
const yaml = require('js-yaml')
const path = require('path')
const leftPad = require('left-pad')

const writeLines = (lines, filePath) => {
  fs.writeFileSync(filePath, lines.join('\n'))
}

/**
 * Write array to gzipped numpy txt format
 */
const npSaveTxt = (array, filePath) => {
  let lines = []
  for (let row of array) {
    if (Array.isArray(row)) {
      // This is a matrix
      lines.push(row.map(it => Number.parseFloat(it).toExponential()).join(' '))
    } else {
      // This is a vector
      lines.push(Number.parseFloat(row).toExponential())
    }
  }
  writeLines(lines, filePath)
}

/**
 * Return a list of paths to model directories
 */
const getModels = (rootDir, live) => {
  let modelsDir = path.join(rootDir, live ? 'model-forecasts/real-time-component-models' : 'model-forecasts/component-models')
  return fs.readdirSync(modelsDir)
    .map(it => path.join(modelsDir, it))
    .filter(it => fs.statSync(it).isDirectory())
    .filter(it => fs.existsSync(path.join(it, 'metadata.txt')))
    .map(it => new Model(it))
}

class Model {
  constructor (modelDir) {
    this.dir = modelDir
    this.meta = yaml.safeLoad(fs.readFileSync(path.join(this.dir, 'metadata.txt'), 'utf8'))
    this.id = `${this.meta.team_name}-${this.meta.model_abbr}`
  }

  get csvs () {
    return fs.readdirSync(this.dir)
      .filter(it => it.endsWith('csv'))
      .sort((a, b) => getCsvEpiweek(a) - getCsvEpiweek(b))
      .map(fileName => path.join(this.dir, fileName))
  }

  getCsvFor (epiweek) {
    let csvs = this.csvs
    return csvs[csvs.map(getCsvEpiweek).findIndex(it => it === epiweek)]
  }
}

/**
 * Return timing information about the csv
 */
const getCsvEpiweek = csvFile => {
  let baseName = path.basename(csvFile)
  let [week, year, ] = baseName.split('-')
  return parseInt(`${year}${leftPad(week.slice(2), 2)}`)
}

/**
 * Return true if all the elements in array are the same
 */
const allSame = (array, comparator) => {
  return array.every((it, i, arr) => comparator(it, arr[0]) === 0)
}

/**
 * Argmax
 */
const argmax = (array, comparator) => {
  let maxIdx = 0
  let maxVal = array[maxIdx]

  for (let i = 1; i < array.length; i++) {
    if (comparator(array[i], maxVal) >= 0) {
      maxIdx = i
      maxVal = array[maxIdx]
    }
  }

  return maxIdx
}

/**
 * Return arrays of indices resulting in the intersection
 * Assume the arrays are sorted
 */
const intersectionIndices = (arrays, comparator) => {
  let indices = arrays.map(it => [])
  let positions = arrays.map(it => 0)

  // Values for current position
  const positionValues = positions => arrays.map((arr, i) => arr[positions[i]])

  // Whether the positions are valid
  const positionsValid = positions => positions.every((p, i) => p < arrays[i].length)

  // March forward
  const advancePositions = positions => {
    // Find the max item, increment all the others
    let values = positionValues(positions)
    let maxIdx = argmax(values, comparator)
    return positions.map((p, i) => comparator(values[i], values[maxIdx]) === 0 ? p : p + 1)
  }

  // Values at the current positions
  let currentValues
  while (positionsValid(positions)) {
    if (allSame(positionValues(positions), comparator)) {
      // If all the elements of current position are equal
      positions.forEach((p, i) => indices[i].push(p))
      positions = positions.map(p => p + 1)
    } else {
      positions = advancePositions(positions)
    }
  }

  return indices
}

/**
 * Python style arange
 */
const arange = (low, high) => {
  return Array(high - low).fill(low).map((it, idx) => it + idx)
}

module.exports.arange = arange
module.exports.getModels = getModels
module.exports.getCsvEpiweek = getCsvEpiweek
module.exports.writeLines = writeLines
module.exports.Model = Model
module.exports.npSaveTxt = npSaveTxt
module.exports.intersectionIndices = intersectionIndices
