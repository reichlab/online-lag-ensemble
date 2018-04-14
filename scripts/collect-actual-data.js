// Script to collect actual flu data

const fct = require('flusight-csv-tools')
const fs = require('fs-extra')
const Papa = require('papaparse')
const u = require('./utils')

if (process.argv.length < 4) {
  console.log('Usage: node <script-name> <index-file> <output-file>')
  process.exit(1)
}

const INDEX_FILE = process.argv[2]
const OUTPUT_FILE = process.argv[3]

/**
 * Return a list of seasons from the range of epiweeks
 */
function getSeasons (epiweeks) {
  let firstSeason = fct.utils.epiweek.seasonFromEpiweek(Math.min(...epiweeks))
  let lastSeason = fct.utils.epiweek.seasonFromEpiweek(Math.max(...epiweeks))
  return u.arange(firstSeason, lastSeason + 1)
}

/**
 * Read in index file
 */
async function parseIndex (indexFile) {
  let fileContent = await fs.readFile(indexFile, 'utf8')
  return Papa.parse(fileContent, { dynamicTyping: true, header: true }).data
}

/**
 * Merge truth values from different seasons in a single object
 */
function mergeTruths (truths) {
  let output = {}
  for (let region of fct.meta.regionIds) {
    output[region] = truths.reduce((acc, t) => acc.concat(t[region]), [])
  }
  return output
}

/**
 * Get truth for all the seasons
 */
async function getTruth (seasons) {
  let lags = u.arange(0, 30) // Only go for 30 lags
  let truths = await Promise.all(lags.map(async (lag) => {
    return {
      lag, seasonValues: await Promise.all(seasons.map(s => fct.truth.getSeasonTruth(s, lag)))
    }
  }))

  return truths
}

/**
 * Write the truth csv to output
 */
async function writeCsv (indexFile, outputFile) {
  let stream = fs.createWriteStream(outputFile, { flags: 'a' })
  let headers = [
    'epiweek',
    'region',
    'lag',
    ...fct.meta.targetIds,
  ]
  stream.write(`${headers.join(',')}\n`)

  let indexData = await parseIndex(indexFile)
  let seasons = getSeasons(indexData.map(d => d.epiweek))
  let truths = await getTruth(seasons)

  for (let { epiweek, region } of indexData) {
    for (let { lag, seasonValues } of truths) {
      let epiweekTruth
      for (let values of seasonValues) {
        epiweekTruth = values[region].find(d => d.epiweek === epiweek)
        if (epiweekTruth) {
          break
        }
      }
      stream.write(`${epiweek},${region},${lag},${fct.meta.targetIds.map(t => epiweekTruth[t] || NaN).join(',')}\n`)
    }
  }
}

writeCsv(INDEX_FILE, OUTPUT_FILE).then(() => console.log('All done'))
