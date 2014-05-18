<?php

function prnt($message) {
	echo "<pre>";
	print_r($message);
	echo "</pre>";
}

$files = glob("../../dev/of/apps/tracking/EmotionReader/bin/data/emotions/*.yml");

$labels = $training_data = "";
$i = $count = 0;

foreach ($files as $f) {

	$yml = file_get_contents($f);
	preg_match_all("~\[(?P<vectors>[^\]]*?)\]~", $yml, $matches);

	foreach ($matches['vectors'] as $v) {
		$labels .= "{$i}, ";
		$training_data .= "{ {$v} }, ";
	}

	$count += count($matches['vectors']);
	$i++;

}

$labels = "float labels[{$count}] = {" . substr($labels, 0, -2) . "};";
$training_data = "float trainingData[{$count}][198] = {" . substr($training_data, 0, -2) . "};";

$result = "{$labels}<br><br>{$training_data};";
file_put_contents("converted.cpp", $result);
echo $result;