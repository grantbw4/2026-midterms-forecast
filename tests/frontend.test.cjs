const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const root = path.join(__dirname, '..');
const source = fs.readFileSync(path.join(root, 'website/js/app.js'), 'utf8');
const html = fs.readFileSync(path.join(root, 'website/index.html'), 'utf8');
const read = name => fs.readFileSync(path.join(root, 'website', name), 'utf8');

// A small DOM adapter exercises the production handlers without external packages.
function dashboard({charts = true} = {}) {
    const nodes = new Map();
    const statuses = [];
    function element() {
        const classes = new Set();
        return {
            dataset: {}, value: '', textContent: '', innerHTML: '', hidden: false, disabled: false,
            children: [], listeners: {}, attributes: {},
            classList: {
                add: name => classes.add(name), remove: name => classes.delete(name),
                toggle(name, active) { active ? classes.add(name) : classes.delete(name); },
                contains: name => classes.has(name),
            },
            setAttribute(name, value) { this.attributes[name] = value; },
            addEventListener(name, callback) { this.listeners[name] = callback; },
            appendChild(child) { this.children.push(child); },
            before(child) { statuses.push(child); },
            getContext() { return {}; },
        };
    }
    for (const match of html.matchAll(/id="([^"]+)"/g)) nodes.set(match[1], element());
    const chamberButtons = ['house', 'senate'].map(chamber => Object.assign(element(), {dataset: {chamber}}));
    const sections = ['house', 'senate'].map(chamber => Object.assign(element(), {dataset: {chamber}}));
    const timelineButtons = ['house', 'senate'].map(timeline => Object.assign(element(), {dataset: {timeline}}));
    const selectors = {'.chamber-btn': chamberButtons, '.chamber-section': sections, '.timeline-btn': timelineButtons};
    const document = {
        getElementById: id => nodes.get(id), createElement: element, addEventListener() {},
        querySelectorAll: selector => selectors[selector] || [],
        querySelector(selector) {
            const status = selector.match(/data-status-for="([^"]+)"/);
            return status ? statuses.find(node => node.dataset.statusFor === status[1]) : element();
        },
    };
    const configs = [];
    const context = vm.createContext({document, console: {warn() {}}, setTimeout, clearTimeout});
    if (charts) context.Chart = class {
        constructor(canvas, config) { configs.push(config); }
        destroy() {}
    };
    vm.runInContext(source, context);
    const run = code => vm.runInContext(code, context);
    context.houseFixture = JSON.parse(read('forecast.json'));
    context.senateFixture = JSON.parse(read('senate_forecast.json'));
    context.houseCsv = read('timeline.csv');
    context.senateCsv = read('senate_timeline.csv');
    run('houseData = houseFixture; senateData = senateFixture; houseTimeline = parseCSV(houseCsv); senateTimeline = parseCSV(senateCsv);');
    return {context, run, nodes, configs, chamberButtons, timelineButtons};
}

test('polling chart preserves all 50 saved observations, ascending dates, and final input', () => {
    const page = dashboard();
    page.run('createNationalTrendChart()');
    const chart = page.configs.at(-1);
    const [history, input] = chart.data.datasets;
    assert.equal(history.data.length, 50);
    assert.equal(chart.data.datasets.length, 2);
    assert.equal(chart.options.scales.x.type, 'linear');
    assert.ok(history.data.every((point, i, all) => !i || point.x > all[i - 1].x));
    assert.equal(history.data.at(-1).x, Date.UTC(2026, 10, 3));
    assert.equal(history.data.at(-1).y, 9.3);
    assert.equal(input.data[0].x, Date.UTC(2026, 10, 2));
    assert.equal(input.data[0].y, 8.68);
    assert.ok(!chart.data.datasets.some(dataset => /band|latent/i.test(dataset.label)));
});

test('timeline preserves actual date gaps and reports the selected chamber change', () => {
    const page = dashboard();
    page.run('createTimelineChart()');
    let chart = page.configs.at(-1);
    const points = chart.data.datasets[0].data;
    assert.equal(points.at(-1).x - points.at(-2).x, 2 * 86400000);
    assert.equal(points.at(-1).y, 91.62);
    assert.equal(chart.options.scales.x.type, 'linear');
    assert.match(page.nodes.get('timeline-stats').textContent, /House.*91.86% on Nov 1, 2026.*91.62% on Nov 3, 2026.*-0.24 percentage points/);
    page.run("currentTimelineChamber = 'senate'; createTimelineChart()");
    chart = page.configs.at(-1);
    assert.equal(chart.data.datasets[0].data.at(-1).y, 76.4);
    assert.match(page.nodes.get('timeline-stats').textContent, /Senate.*74.08%.*76.40%.*\+2.32 percentage points/);
});

test('Senate distribution requires 51 Democratic seats for control', () => {
    const page = dashboard();
    page.run("currentChamber = 'senate'; createSeatChart()");
    const chart = page.configs.at(-1);
    const colors = chart.data.datasets[0].backgroundColor;
    assert.equal(colors[chart.data.labels.indexOf(50)], 'rgba(188, 0, 0, 0.8)');
    assert.equal(colors[chart.data.labels.indexOf(51)], 'rgba(0, 21, 188, 0.8)');
});

test('missing Chart.js and Leaflet leave stats, table, filters, and modals usable', () => {
    const page = dashboard({charts: false});
    page.run('initializePage()');
    assert.equal(page.nodes.get('showing-count').textContent, 435);
    assert.equal(page.nodes.get('house-dem-prob').textContent, '92%');
    assert.ok(page.nodes.get('district-tbody').innerHTML.includes('AL-02'));
    assert.equal(page.nodes.get('seat-chart').hidden, true);
    page.nodes.get('category-filter').value = 'safe_d';
    page.nodes.get('search-filter').value = 'AL-02';
    page.chamberButtons[1].listeners.click();
    assert.equal(page.nodes.get('category-filter').value, '');
    assert.equal(page.nodes.get('search-filter').value, '');
    assert.equal(page.nodes.get('showing-count').textContent, 35);
    assert.match(page.nodes.get('chart-stats').innerHTML, /76%/);
    page.run("showDistrictModal('AK')");
    assert.equal(page.nodes.get('district-modal').classList.contains('active'), true);
    page.run('closeModal()');
    assert.equal(page.nodes.get('district-modal').classList.contains('active'), false);
});

test('a failed chart constructor cannot prevent core initialization or chamber switching', () => {
    const page = dashboard();
    page.context.Chart = class { constructor() { throw new Error('Unavailable canvas'); } };
    page.run('initializePage()');
    assert.equal(page.nodes.get('showing-count').textContent, 435);
    page.chamberButtons[1].listeners.click();
    assert.equal(page.nodes.get('showing-count').textContent, 35);
});

test('missing chamber or timeline data never displays the other chamber as a substitute', () => {
    const page = dashboard();
    page.run('houseData = null; senateTimeline = null; initializePage()');
    assert.equal(page.chamberButtons[0].disabled, true);
    assert.equal(page.nodes.get('showing-count').textContent, 35);
    page.timelineButtons[1].listeners.click();
    assert.equal(page.nodes.get('timeline-chart').hidden, true);
    assert.match(page.nodes.get('timeline-stats').textContent, /Two saved forecasts/);
});

test('CSV handles CRLF and dates and timestamps remain timezone-stable', () => {
    const page = dashboard();
    assert.equal(page.run("parseCSV('date,value\\r\\n2026-11-03,1\\r\\n')[0].date"), '2026-11-03');
    assert.equal(page.run("chartDate(dateValue('2026-11-03'))"), 'Nov 3');
    assert.match(page.run("formatDate(new Date('2026-11-03T08:16:10Z'))"), /November 3, 2026.*12:16.*PST/);
});
